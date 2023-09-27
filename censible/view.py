"""This module provides the `View` utility classfor reshaping PyTorch tensors.

It is built on top of the PyTorch nn.Module, making it integrable within any
standard PyTorch model.
"""

import torch.nn as nn


class View(nn.Module):
    """A class for reshaping tensors."""

    def __init__(self, shape):
        """Initialize the View.

        Args:
            shape (tuple): The shape to reshape the tensor to.
        """
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        """Reshape the input tensor to the specified shape.
        
        Args:
            input (torch.Tensor): The input tensor.
            
        Returns:
            The reshaped tensor.
        """
        return input.view(*self.shape)
