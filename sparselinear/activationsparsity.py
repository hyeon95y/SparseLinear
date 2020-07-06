import math
import torch
import torch.nn as nn

class ActivationSparsity(nn.Module):
    """Applies activation sparsity to the last dimension of input using K-winners strategy

    Args:
        alpha (float): constant used in updating duty-cycle
            Default: 0.1
        beta (float): boosting factor for neurons not activated in the previous duty cycle
            Default: 1.5
        act_sparsity (float): fraction of the input used in calculating K for K-Winners strategy
            Default: 0.65
    
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input
        
    Examples::
    
        >>> x = asy.ActivationSparsity(10)
        >>> input = torch.randn(3,10)
        >>> output = x(input)
    """
    def __init__(self, alpha=0.1, beta=1.5, act_sparsity=0.65):
        super(ActivationSparsity, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.act_sparsity = act_sparsity
        self.duty_cycle = None
    
    def updateDC(self, inputs, duty_cycle):
        duty_cycle = (1 - self.alpha) * duty_cycle + self.alpha * (inputs.gt(0).sum(dim=0,dtype=torch.float))
        return duty_cycle

    def forward(self, inputs):
        in_features = inputs.shape[-1]
        out_shape=list(inputs.shape)
        inputs = inputs.reshape(inputs.shape[0],-1)

        device = inputs.device
       
        if self.duty_cycle is None:
            self.duty_cycle = torch.zeros(in_features, requires_grad=True).to(device)
        
        k = math.floor((1-self.act_sparsity) * in_features)
        with torch.no_grad():
            
            target = k / inputs.shape[-1]
            boost_coefficient = torch.exp(self.beta * (target - self.duty_cycle))
            boosted_input = inputs * boost_coefficient 
            
            # Get top k values 
            values, indices = boosted_input.topk( k, dim=-1, sorted=False)
            row_indices = torch.arange(inputs.shape[0]).repeat_interleave(k).view(-1,k)
            
        outputs = torch.zeros_like(inputs).to(device)
        outputs = outputs.index_put((row_indices, indices), inputs[row_indices, indices], accumulate=False) 
        
        if self.training:
            with torch.no_grad():
                self.duty_cycle = self.updateDC(outputs, self.duty_cycle)
        
        return outputs.view(out_shape)
    
    def extra_repr(self):
        return 'act_sparsity={}, alpha={}, beta={}, duty_cycle={}'.format(
            self.act_sparsity, self.alpha, self.beta, self.duty_cycle
        )