import numpy as np
import torch.nn as nn


def ppo_layer_init(layer, std=np.sqrt(2), bias_constant=0.0):
    """Initialize layer weights using the PPO initialization method.

    This function initializes the layer weights by drawing samples 
    from a Gaussian distribution with a mean of 0 and a standard 
    deviation of std. The biases are initialized with a constant 
    value of bias_constant. Moreover, this function applies orthogonal 
    initialization to the weight matrix, ensuring that the weights 
    are initialized in a way that preserves the orthogonality of the
    weight vectors. This is done because PPO requires orthogonal 
    weights to work and PyTorch does not do that by default.

    Args:
        layer (torch.nn.Module): The layer to initialize.
        std (float, optional): The standard deviation of the weight initialization.
            Defaults to sqrt(2).
        bias_constant (float, optional): The constant value for bias initialization.
            Defaults to 0.0.

    Returns:
        torch.nn.Module: The initialized layer.
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_constant)
    return layer
