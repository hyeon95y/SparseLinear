# SparseLinear

SparseLinear is a pytorch package that allows a user to create extremely wide and sparse linear layers efficiently. A sparsely connected network is a network where each node is connected to a fraction of available nodes. This differs from a fully connected network, where each node in one layer is connected to every node in the next layer.

The provided layer along with the dynamic activation sparsity module is compatible with backpropagation. The sparse linear layer is initialized with sparsity, supports unstructured sparsity and allows dynamic growth and pruning. We achieve this by building a linear layer on top of [PyTorch Sparse](https://github.com/rusty1s/pytorch_sparse), which provides optimized sparse matrix operations with autograd support in PyTorch.

## Table of Contents

- [More about SparseLinear](#intro)
- [More about Dynamic Activation](#kwin)
- [Installation](#install)
- [Getting Started](#usage)
- [Benchmark Results](#results)

### More about SparseLinear <a name="intro"></a>
The default arguments initialize a sparse linear layer with random connections that applies a linear transformation to the incoming data `y = xA^T + b` 

##### Parameters

- **in_features** - size of each input sample
- **out_features** - size of each output sample
- **bias** - If set to ``False``, the layer will not learn an additive bias. Default: ``True``
- **sparsity** - sparsity of weight matrix. Default: 0.9
- **connectivity** - User defined sparsity matrix. Default: None
- **small_world** - Boolean flag to generate small world sparsity. Default: ``False``
- **dynamic** - Boolean flag to dynamically change the network structure. Default: ``False``
- **deltaT** - frequency for growing and pruning update step. Default: 6000
- **Tend** - Stopping time for growing and pruning algorithm update step. Default: 150000
- **alpha** - f-decay parameter for cosine updates. Default: 0.1
- **max_size** - Maximum number of entries allowed before chunking occurrs. Default: 1e8

##### Shape

- Input: `(N, *, H_{in})` where :math:`*` means any number of additional dimensions and `H_{in} = \text{in\_features}`
- Output: `(N, *, H_{out})` where all but the last dimension are the same shape as the input and `H_{out} = \text{out\_features}`.

##### Variables 

- **~SparseLinear.weight** - the learnable weights of the module of shape `(\text{out\_features}, \text{in\_features})`. The values are initialized from `\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where `k = \frac{1}{\text{in\_features}}`
- **bias** - the learnable bias of the module of shape `(\text{out\_features})`. If :attr:`bias` is ``True``, the values are initialized from `\mathcal{U}(-\sqrt{k}, \sqrt{k})` where `k = \frac{1}{\text{in\_features}}`

##### Examples:

```python
 >>> m = nn.SparseLinear(20, 30)
 >>> input = torch.randn(128, 20)
 >>> output = m(input)
 >>> print(output.size())
 torch.Size([128, 30])
```

The following customization can also be done using appropriate arguments -

#### User-defined Sparsity

One can choose to add self-defined static sparsity. The `connectivity` flag accepts a (2, nnz) LongTensor that represents the rows and columns of nonzero elements in the layer. 

#### Small-world Sparsity

The default static sparsity is random. With this flag, one can instead use small-world sparsity. See [here](https://en.wikipedia.org/wiki/Small-world_network). To specify, set `small_world` to `True`. Specifically, we make connections distant dependent to ensure small-world behavior.

#### Dynamic Growing and Pruning Algorithm

The user can grow and prune units during training starting from a sparse configuration using this feature. The implementation is based on [Rigging the lottery](https://arxiv.org/pdf/1911.11134.pdf) algorithm. Specify `dynamic` to be `True` to dynamically alter the layer connections while training. 

## Dynamic Activation Sparsity <a name="kwin"></a>

In addition, we provide a Dynamic Activation Sparsity module to utilize principled, per-layer activation sparsity. The algorithm implementation is based on the [K-Winners strategy](https://arxiv.org/pdf/1903.11257.pdf). 

### Installation <a name="install"></a>

### Getting Started <a name="usage"></a>

### Benchmark Results <a name="results"></a>
