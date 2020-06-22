# SparseLinear

SparseLinear is a pytorch package that allows a user to create extremely wide and sparse linear layers efficiently. 

It has the following functionalities available at the moment - 

## User-defined Sparsity

One can choose to add a self-defined static sparsity. This involves adding an argument that accepts a (2, nnz) (or maybe (nnz, 2)) tensor that represents the rows and columns of nonzero elements.

## Small-world Sparsity

The default static sparsity is random. With this flag, the can instead use small-world sparsity. See [here](https://en.wikipedia.org/wiki/Small-world_network)

## Dynamic Activation Sparsity

The user has an option to utilize principled, per-layer activation sparsity. The algorithm implementation is based on the [K-Winners strategy](https://arxiv.org/pdf/1903.11257.pdf). 

## Dynamic Growing and Pruning Algorithm

The user can grow and prune units during training starting from a sparse configuration using this feature. The implementation is based on [Rigging the lottery](https://arxiv.org/pdf/1911.11134.pdf) algorithm.
