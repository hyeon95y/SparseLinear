import warnings
import numpy as np
import math

import torch
import torch.nn as nn

import torch_sparse

# Custom pytorch function to handle growing connections
class GrowConnections(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, k, indices, features, max_size):

        out_features, in_features = features
        
        output_shape = list(inputs.shape)
        output_shape[-1] = out_features
        if len(output_shape) == 1: inputs = inputs.view(1, -1)
        inputs = inputs.flatten(end_dim=-2)

        output = torch_sparse.spmm(indices, weights, out_features, in_features, inputs.t()).t()
        
        ctx.save_for_backward(inputs, weights, indices)
        ctx.in1 = k
        ctx.in2 = out_features
        ctx.in3 = in_features
        ctx.in4 = max_size

        return output

    @staticmethod
    def backward(ctx, grad_output):

        inputs, weights, indices  = ctx.saved_tensors
        k = ctx.in1
        out_features = ctx.in2
        in_features = ctx.in3
        max_size = ctx.in4

        device = grad_output.device
        p_index = torch.LongTensor([1,0])
        new_indices = torch.zeros_like(indices).to(device=device)
        new_indices[p_index] = indices
        
        grad_input = torch_sparse.spmm(new_indices, weights, in_features, out_features, grad_output.t()).t()

        if in_features*out_features <= max_size:
            grad_weights = torch.abs(torch.matmul(inputs.t(),grad_output).t())
            mask = torch.ones_like(grad_weights)
            mask[indices[0], indices[1]] =  0
            
            _, lm_indices = torch.topk((mask*grad_weights).reshape(-1), k, sorted=False)
            row = lm_indices.floor_divide(in_features)
            col = lm_indices.fmod(in_features)

        else:
            tk = None
            m = max_size/in_features
            chunks = math.ceil(out_features/m)
            
            for item in range(chunks):
                if item!=chunks-1:
                    grad_m = torch.matmul(inputs.t()[item*m:(item+1)*m,:], grad_output).t()
                    grad_m_abs = torch.abs(grad_m)
                    topk_values, topk_indices = torch.topk(grad_m_abs.view(-1), k, sorted=False)
                else:
                    grad_m = torch.matmul(inputs.t()[item*m:,:], grad_output).t()
                    grad_m_abs = torch.abs(grad_m)
                    topk_values, topk_indices = torch.topk(grad_m_abs.view(-1), k, sorted=False)
                
                row = topk_indices.floor_divide(in_features) + torch.ones_like(topk_indices)*item*m
                col = topk_indices.fmod(in_features)
                indices = torch.stack((row,col))
                
                if tk is None:
                    tk = torch.cat((topk_values, indices), dim=0)
                else:
                    topk_values_prev = tk[0]
                    topk_values_2k, topk_indices_2k = torch.topk(torch.cat((topk_values_prev, topk_values), dim=1).view(-1),k,sorted=False)
                    
                    # Get the topk indices from the combination of two indices
                    topk_prev = topk_indices_2k[topk_indices_2k<k]
                    topk_values_indices = tk[:][topk_prev]

                    topk_curr = topk_indices_2k[topk_indices_2k>=k]
                    topk_curr = topk_curr % k

                    curr_indices = indices[:][topk_curr]
                    curr_values = topk_values[topk_curr]

                    tk = torch.cat((topk_values_indices,(torch.cat((curr_indices,curr_values),dim=0))),dim=1)
            row = tk[1]
            col = tk[2]

        new_indices = torch.stack((row, col))
        x = torch.cat((indices[:,:-k], new_indices),dim=1)

        if indices.shape[1] > x.shape[1]:
            diff = indices.shape[1] - x.shape[1] 
            x = torch.cat((x,torch.zeros((2, diff), dtype=torch.long).to(device=device)), dim=1)

        indices.copy_(x)

        return grad_input, None, None, None, None, None


class SparseLinear(nn.Module):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        sparsity: sparsity of weight matrix
            Default: 0.9
        connectivity: user defined sparsity matrix
            Default: None
        small_world: boolean flag to generate small world sparsity
            Default: ``False``
        dynamic: boolean flag to dynamically change the network structure
            Default: ``False``
        deltaT (int): frequency for growing and pruning update step
            Default: 6000
        Tend (int): stopping time for growing and pruning algorithm update step
            Default: 150000
        alpha (float): f-decay parameter for cosine updates
            Default: 0.1
        max_size (int): maximum number of entries allowed before chunking occurrs
            Default: 1e8
    
    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    
    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
    
    Examples::
        
        >>> m = nn.SparseLinear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    def __init__(self, in_features, out_features, bias=True, sparsity=0.9, connectivity=None, small_world=False, dynamic=False, deltaT=6000, Tend=150000, alpha=0.1, max_size=1e8):
        assert in_features < 2**31 and out_features < 2**31 and sparsity < 1.0
        assert connectivity is None or not small_world, "Cannot specify connectivity along with small world sparsity"
        if connectivity is not None:
            assert isinstance(connectivity, torch.LongTensor) or isinstance(connectivity, torch.cuda.LongTensor), "Connectivity must be a Long Tensor"
            assert connectivity.shape[0]==2 and connectivity.shape[1]>0, "Input shape for connectivity should be (2,nnz)"
            assert connectivity.shape[1] <= in_features*out_features, "Nnz can't be bigger than the weight matrix"
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.connectivity = connectivity
        self.small_world = small_world
        self.dynamic = dynamic
        self.max_size = max_size
        
        # Generate and coalesce indices
        coalesce_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # Faster to coalesce on GPU
        if not small_world:
            if connectivity is None:
                self.sparsity = sparsity
                nnz = round((1.0-sparsity) * in_features * out_features)
                if in_features * out_features <= 10**8:
                    indices = np.random.choice(in_features * out_features, nnz, replace=False)
                    indices = torch.as_tensor(indices, device=coalesce_device)
                    row_ind = indices.floor_divide(in_features)
                    col_ind = indices.fmod(in_features)
                else:
                    warnings.warn("Matrix too large to sample non-zero indices without replacement, sparsity will be approximate", RuntimeWarning)
                    row_ind = torch.randint(0, out_features, (nnz,), device=coalesce_device)
                    col_ind = torch.randint(0, in_features, (nnz,), device=coalesce_device)
                indices = torch.stack((row_ind, col_ind))
            else:
                # User defined sparsity
                nnz = connectivity.shape[1]
                self.sparsity = nnz/(out_features*in_features)
                connectivity = connectivity.to(device=coalesce_device)
                indices = connectivity
                
        else:
            #Generate small world sparsity
            self.sparsity = sparsity
            nnz = round((1.0-sparsity) * in_features * out_features)
            assert nnz > min(in_features, out_features), 'The matrix is too sparse for small-world algorithm; please decrease sparsity'
            offset = abs(out_features - in_features) / 2.

            # Node labels
            inputs = torch.arange(1 + offset * (out_features > in_features), in_features + 1 + offset * (out_features > in_features), device=coalesce_device)
            outputs = torch.arange(1 + offset * (out_features < in_features), out_features + 1 + offset * (out_features < in_features), device=coalesce_device)

            total_data = in_features * out_features                 # Total params
            chunks = math.ceil(total_data / self.max_size)
            split_div = max(in_features, out_features) // chunks    # Full chunks
            split_mod = max(in_features, out_features) % chunks     # Remaining chunk
            idx = torch.repeat_interleave(torch.Tensor([split_div]), chunks).int().to(device=coalesce_device)
            idx[:split_mod] += 1
            idx = torch.cumsum(idx, dim=0)
            idx = torch.cat([torch.LongTensor([0]).to(device=coalesce_device), idx])

            count = 0

            rows = torch.empty(0).long().to(device=coalesce_device)
            cols = torch.empty(0).long().to(device=coalesce_device)

            def small_world_chunker(inputs, outputs, nnz):
                pair_distance = inputs.view(-1, 1) - outputs
                arg = torch.abs(pair_distance) + 1.
                # lambda search
                error = float('inf')
                L, U = 1e-5, 5.  
                lamb = 1.                   # initial guess
                itr = 1
                error_threshold = 10.
                max_itr = 1000
                P = arg**(-lamb)
                P_sum = P.sum()
                error = abs(P_sum - nnz)

                while error > error_threshold:
                    assert itr <= max_itr, 'No solution found; please try different network sizes and sparsity levels'
                    if P_sum < nnz:
                        U = lamb
                        lamb = (lamb + L) / 2.
                    elif P_sum > nnz:
                        L = lamb
                        lamb = (lamb + U) / 2.
                        
                    P = arg**(-lamb)
                    P_sum = P.sum()
                    error = abs(P_sum - nnz)
                    itr += 1
                return P

            for i in range(chunks):
                inputs_ = inputs[idx[i]:idx[i+1]] if out_features < in_features else inputs
                outputs_ = outputs[idx[i]:idx[i+1]] if out_features > in_features else outputs

                y = small_world_chunker(inputs_, outputs_, round(nnz / chunks))
                ref = torch.rand_like(y)
                
                mask = torch.empty(y.shape, dtype=bool).to(device=coalesce_device)
                mask[y < ref] = False
                mask[y >= ref] = True

                rows_, cols_ = mask.to_sparse().indices()

                rows = torch.cat([rows, rows_ + idx[i]])
                cols = torch.cat([cols, cols_])

            indices = torch.stack((cols, rows))
            nnz = indices.shape[1]

        values = torch.empty(nnz, device=coalesce_device)

        indices, values = torch_sparse.coalesce(indices, values, out_features, in_features)

        
        self.register_buffer('indices', indices.cpu())
        self.weights = nn.Parameter(values.cpu())

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        if self.dynamic:
            self.deltaT = deltaT
            self.Tend = Tend
            self.alpha = alpha
            self.itr_count = 0

        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / self.in_features**0.5
        nn.init.uniform_(self.weights, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def weight(self):
        """ returns a torch.sparse.FloatTensor view of the underlying weight matrix 
            This is only for inspection purposes and should not be modified or used in any autograd operations
        """
        weight = torch.sparse.FloatTensor(self.indices, self.weights, (self.out_features, self.in_features))
        return weight.coalesce().detach()

    def forward(self, inputs):
        if self.dynamic:
            self.itr_count+= 1
        output_shape = list(inputs.shape)
        output_shape[-1] = self.out_features

        # Handle dynamic sparsity
        if self.training and self.dynamic and self.itr_count < self.Tend and self.itr_count%self.deltaT==0:
            
            #Drop criterion
            f_decay = self.alpha * (1 + math.cos(self.itr_count * math.pi/self.Tend))/2
            k = int(f_decay *( 1 - self.sparsity ) * self.weights.view(-1,1).shape[0])
            n = self.weights.shape[0]
            
            _, lm_indices = torch.topk(-torch.abs(self.weights),n-k, largest=False, sorted=False)

            self.indices = torch.index_select(self.indices,1, lm_indices)
            self.weights = nn.Parameter(torch.index_select(self.weights, 0, lm_indices))

            device = inputs.device
            #Growth criterion
            self.weights = nn.Parameter(torch.cat((self.weights,((torch.zeros(k))).to(device=device)),dim=0))
            self.indices = torch.cat((self.indices,torch.zeros((2,k), dtype=torch.long).to(device=device)),dim=1)
            output = GrowConnections.apply( inputs, self.weights, k, self.indices, (self.out_features, self.in_features), self.max_size)

        else:

            if len(output_shape) == 1: inputs = inputs.view(1, -1)
            inputs = inputs.flatten(end_dim=-2)

            output = torch_sparse.spmm(self.indices, self.weights, self.out_features, self.in_features, inputs.t()).t()
            if self.bias is not None:
                output += self.bias
 
        return output.view(output_shape)
            

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, sparsity={}, connectivity={}, small_world={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.sparsity, self.connectivity, self.small_world
        )
