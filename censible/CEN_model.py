"""
This module defines the neural network architecture.

The network is a convolutional architecture tailored for molecular data, which,
in its forward pass, computes both predicted affinities and the coefficients for
precalculated molecular terms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from censible.view import View

# Compare to published performance for this model, 1.5 RMSE and 0.7 Pearson's R,
# when predicting affinity alone.


class CENet(nn.Module):
    """Default 2018, but final layer generates coefficients for terms."""

    def __init__(self, dims, numterms=None):
        """Initialize the model.
        
        Args:
            dims (list): A list of integers representing the dimensions of the
                input.
            numterms (int): The number of terms to use. Can also be None.
        """
        super(CENet, self).__init__()
        self.modules = []
        nchannels = dims[0]
        self.nterms = numterms
        self.func = F.relu

        avgpool1 = nn.AvgPool3d(2, stride=2)
        self.add_module("avgpool_0", avgpool1)
        self.modules.append(avgpool1)
        conv1 = nn.Conv3d(
            nchannels, out_channels=32, padding=1, kernel_size=3, stride=1
        )
        self.add_module("unit1_conv", conv1)
        self.modules.append(conv1)
        conv2 = nn.Conv3d(32, out_channels=32, padding=0, kernel_size=1, stride=1)
        self.add_module("unit2_conv", conv2)
        self.modules.append(conv2)
        avgpool2 = nn.AvgPool3d(2, stride=2)
        self.add_module("avgpool_1", avgpool2)
        self.modules.append(avgpool2)
        conv3 = nn.Conv3d(32, out_channels=64, padding=1, kernel_size=3, stride=1)
        self.add_module("unit3_conv", conv3)
        self.modules.append(conv3)
        conv4 = nn.Conv3d(64, out_channels=64, padding=0, kernel_size=1, stride=1)
        self.add_module("unit4_conv", conv4)
        self.modules.append(conv4)
        avgpool3 = nn.AvgPool3d(2, stride=2)
        self.add_module("avgpool_2", avgpool3)
        self.modules.append(avgpool3)
        conv5 = nn.Conv3d(64, out_channels=128, padding=1, kernel_size=3, stride=1)
        self.add_module("unit5_conv", conv5)
        self.modules.append(conv5)
        div = 2 * 2 * 2
        last_size = int(dims[1] // div * dims[2] // div * dims[3] // div * 128)
        # print(last_size)
        flattener = View((-1, last_size))
        self.add_module("flatten", flattener)
        self.modules.append(flattener)
        self.fc = nn.Linear(last_size, numterms)
        self.add_module("last_fc", self.fc)

    def forward(self, batch: torch.Tensor, precalculated_terms: torch.Tensor) -> tuple:
        """Forward pass.
        
        Args:
            batch (torch.Tensor): A torch tensor representing a batch of data.
            precalculated_terms (torch.Tensor): A torch tensor representing the
                precalculated terms.
        
        Returns:
            A tuple of torch tensors representing the predicted affinity, the
            predicted coefficients, and the weighted terms.
        """
        for layer in self.modules:
            batch = layer(batch)
            if isinstance(layer, nn.Conv3d):
                batch = self.func(batch)
        # coef_predict = self.fc(x) / 1000  # JDD added
        coef_predict = self.fc(batch)
        batch_size, num_terms = coef_predict.shape

        # Here also predict term * weight for each term. For another graph, that
        # isn't scaled.
        # coef_predict = coef_predict.view(batch_size, num_terms, -1)

        # Do batchwise, pairwise multiplication coef_predict and smina_terms
        weighted_terms = coef_predict * precalculated_terms

        # batchwise dot product
        return (
            torch.bmm(
                coef_predict.view(batch_size, 1, num_terms),
                precalculated_terms.view(batch_size, num_terms, 1),
            ),
            coef_predict,
            weighted_terms,
        )
