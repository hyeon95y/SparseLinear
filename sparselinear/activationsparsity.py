import math
import torch
import torch.nn as nn

class ActivationSparsity(nn.Module):
    """Applies activation sparsity using k-winners strategy to the input.
    See here: https://arxiv.org/abs/1903.11257

    Args:
        in_features (int): size of each input sample
        k (int): The K in K-Winners strategy
        alpha (float): The constant used in updating duty-cycle
        beta (float): Boosting factor for neurons not activated in the previous duty cycle
        act_sparsity (float): Proportion of innactive units
    Shape:
        Input: (N, in_features) 
        Output: (N, in_features)
    Examples:
        >>> input = torch.randn(3, 10)
        >>> x = asy.ActivationSparsity(10)
        >>> output = x(input) 
    """
    def __init__(self, k=5, alpha=0.1, beta=1, act_sparsity=0.8):
        super(ActivationSparsity, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.prev_duty_cycle = None
        self.act_sparsity = act_sparsity
        self.k = None
    
    def updateDC(self, inputs, prev_duty_cycle):
        """Function to calculate the activation sparsity using the k-winners strategy. This is
            to be used as an activation function by the sparselinear layer. 

        Args:
            inputs ([torch.Tensor]): [Input tensor from the linear layer]
            k (int, optional): [The value of k in k-winners strategy]. Defaults to 5.
            alpha (float, optional): [Contribution factor of top indices to the duty cycle]. Defaults to 0.1.
            beta (float, optional): [Boost coefficient for units in a layer]. Defaults to 0 i.e. No boosting. 
        """
        duty_cycle = (1 - self.alpha) * self.prev_duty_cycle + self.alpha * inputs
        return duty_cycle

    def forward(self, inputs):
        out_shape = list(inputs.shape)
        inputs = inputs.reshape(inputs.shape[0], -1)

        if self.prev_duty_cycle is None:
            self.prev_duty_cycle = torch.zeros(inputs.shape[-1])
        
        if self.k is None:
            self.k = math.floor(self.act_sparsity * inputs.shape[-1])
        
        target = self.k / torch.norm(inputs, dim=-1, keepdim=True)
        boost_coefficient = torch.exp(self.beta * (target - self.prev_duty_cycle))
        
        values, indices = torch.topk(boost_coefficient * inputs, self.k, dim=-1, sorted=False)
        outputs = torch.zeros_like(inputs).t().scatter_(0, indices.t(), values.t()).t()
        
        if self.training:
            self.prev_duty_cycle = self.updateDC(outputs, self.prev_duty_cycle)
        
        return outputs.view(out_shape)
    
    def extra_repr(self):
        return 'k={}, alpha={}, beta={}, prev_duty_cycle={}'.format(
            self.k, self.alpha, self.beta, self.prev_duty_cycle
        )
