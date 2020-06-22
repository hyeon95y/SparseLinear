# SparseLinear

SparseLinear is a pytorch package that allows a user to create extremely wide and sparse linear layers efficiently. A sparsely connected network is a network where each node is connected to only a smaller fraction of available nodes. This differs from a fully connected network, where each node in one layer is connected to every node in the next layer.

The provided layer is compatible with backpropagation, is initialized with sparsity, and supports unstructured sparsity. We achieve this by building a linear layer on top of PyTorch Sparse, which provides optimized sparse matrix operations with autograd support in PyTorch.

The default arguments initialise a sparse linear layer with random connections. The following customization can be done using appropriate arguments -

### User-defined Sparsity

One can choose to add a self-defined static sparsity. The `connectivity` flag accepts a (2, nnz) LongTensor that represents the rows and columns of nonzero elements in the layer. 

### Small-world Sparsity

The default static sparsity is random. With this flag, one can instead use small-world sparsity. See [here](https://en.wikipedia.org/wiki/Small-world_network). To specify, set `small_world` to `True`. 

### Dynamic Growing and Pruning Algorithm

The user can grow and prune units during training starting from a sparse configuration using this feature. The implementation is based on [Rigging the lottery](https://arxiv.org/pdf/1911.11134.pdf) algorithm. Specify `dynamic` to be `True` to dynamically alter the layer connections while training. 

In addition, we provide a Dynamic Activation Sparsity module to utilize principled, per-layer activation sparsity. 

## Dynamic Activation Sparsity

The algorithm implementation is based on the [K-Winners strategy](https://arxiv.org/pdf/1903.11257.pdf). 
