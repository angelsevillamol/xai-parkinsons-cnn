#!/usr/bin/env python3

"""Network Architectures

All the network architectures are defined in this file as PyTorch
nn.Module's, as well as the loss function used for the ordinal
methodology.

- ConvNet: Convolutional part, common part of both nominal and ordinal
    architectures
- NominalDenseNet, OrdinalDenseNet: hidden fully connected parts of the
    nominal and ordinal architectures, respectively.
- NominalNet, OrdinalNet: Combination of convolutional and hidden fully
    connected parts of the nominal and ordinal architectures
    (ConvNet + NominalDenseNet, ConvNet + OrdinalDenseNet)
- ordinal_distance_loss: ordinal loss function

Author: AYRNA
source: https://github.com/ayrna/ordinal-cnn-parkinsons/blob/main/network.py
Modified by: Angel Sevilla Molina
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cdist


def conv_output_shape(
    input_shape : np.ndarray,
    kernel_size : int,
    stride : int,
    padding : int = 0,
    dilation : int = 1
) -> int:
    """Computes output shape of convolution.

    Args:
      input_shape: the shape of the input image.
      kernel_size: the size of the kernel.
      stride: the stride of the convolution.
      padding: the padding added to the input.
      dilation: the spacing between kernel elements.

    Returns:
      The output shape of the convolution.
    """
    return np.floor(
        (
            (input_shape + 2*padding - dilation * (kernel_size-1) - 1) / stride
        ) + 1
    )


class BrainNet(nn.Module):
    """
    Base class for neural network architectures.
    """
    convnet: nn.Module
    densenet: nn.Module

    def predict(self, x : torch.Tensor) -> torch.Tensor:
        """Makes a prediction from input data.

        Args:
          x: the input data tensor.

        Returns:
          A tensor with the network predictions.
        """
        self.eval()
        x = self.convnet(x)
        return self.densenet.predict(x)

    def outputs(self, x : torch.Tensor) -> torch.Tensor:
        """Computes network outputs.

        Args:
          x: the input data tensor

        Returns:
          The outputs from the dense part of the network.
        """
        self.eval()
        x = self.convnet(x)
        return self.densenet.outputs(x)


class ConvNet(nn.Module):
    """
    Convolutional part, common part of both nominal and ordinal
    architectures.
    """
    def __init__(
        self,
        image_shape : Tuple[int, ...],
        n_channels : List[int],
        kernel_size : int,
        stride : int
    ):
        """Initializes the convolutional part of a neural network.

        Args:
          image_shape: the shape of the input image.
          n_channels: the number of input channels.
          kernel_size: the size of the kernel.
          stride: the stride of the convolution.
        """
        super(ConvNet, self).__init__()

        self.n_channels = n_channels
        inout = zip(self.n_channels[:-1], self.n_channels[1:])
        self.convs = list()
        shape = np.array(image_shape)
        for in_channels, out_channels in inout:
            conv_layer = nn.Conv3d(in_channels, out_channels,
                                   kernel_size, stride)
            shape = conv_output_shape(shape, kernel_size, stride)

            self.convs.append(conv_layer)
        self.batch_norms = [nn.BatchNorm3d(nc, eps=1e-3, momentum=0.01)
                            for nc in self.n_channels[1:]]

        self.convs = nn.ModuleList(self.convs)
        self.batch_norms = nn.ModuleList(self.batch_norms)

        self.conv_output_size = self.n_channels[-1] * np.prod(shape).astype(int)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """Performs forward propagation of input data through the
        network.

        Args:
          x: the input data tensor

        Returns:
          The output from the convolutional part of the network.
        """
        for conv, bn in zip(self.convs, self.batch_norms):
            x = self.leaky_relu(bn(conv(x)))

        x = x.view(-1, self.conv_output_size)
        return x


class NominalDenseNet(nn.Module):
    """
    Hidden fully connected parts of the nominal architecture.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_classes: int,
        dropout_rate: float
    ):
        """Initializes the fully connected part of a nominal
        classification neural network.

        Args:
          input_size: the size of the input.
          hidden_size: the size of the hidden layer.
          n_classes: the number of output classes.
          dropout_rate: the dropout rate.
        """
        super(NominalDenseNet, self).__init__()
        self.dense_hidden = nn.Linear(input_size, hidden_size)
        self.dense_hidden_dropout = nn.Dropout(dropout_rate)
        self.dense_output = nn.Linear(hidden_size, n_classes)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """Performs forward propagation of input data through the
        network.

        Args:
          x: the input data tensor

        Returns:
          The output from the dense part of the network. The output is
          interpreted as the probability of belonging to the classes.
        """
        x = self.leaky_relu(self.dense_hidden(x))
        x = self.dense_hidden_dropout(x)
        x = self.dense_output(x)
        return x

    def predict(self, x : torch.Tensor) -> torch.Tensor:
        """Makes a prediction from input data.

        Args:
          x: the input data tensor

        Returns:
          A tensor with the network predictions.
        """
        self.eval()
        outputs = self.outputs(x)
        labels = outputs.argmax(axis=1)
        return labels

    def outputs(self, x : torch.Tensor) -> np.ndarray:
        """Computes network outputs.

        Args:
          x: the input data tensor

        Returns:
          The outputs from the dense part of the network.
        """
        return self.forward(x).detach().cpu().numpy()


class NominalNet(BrainNet):
    """
    Combination of convolutional and hidden fully connected parts of
    the nominal architecture (ConvNet + NominalDenseNet).
    """
    def __init__(
        self,
        image_shape : Tuple[int, ...],
        n_channels : List[int],
        kernel_size : int,
        stride : int,
        hidden_size : int,
        n_classes : int,
        dropout_rate : float
    ):
        """Initializes a nominal classification convolutional neural
        network.

        Args:
          image_shape: the shape of the input image.
          n_channels: the number of input channels.
          kernel_size: the size of the kernel.
          stride: the stride of the convolution.
          hidden_size: the size of the hidden layer.
          n_classes: the number of output classes.
          dropout_rate: the dropout rate.
        """
        super(NominalNet, self).__init__()
        self.convnet = ConvNet(image_shape, n_channels, kernel_size, stride)
        self.densenet = NominalDenseNet(self.convnet.conv_output_size,
                                        hidden_size, n_classes, dropout_rate)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """Performs forward propagation of input data through the
        network.

        Args:
          x: the input data tensor

        Returns:
          The output from the dense part of the network. The output is
          interpreted as the probability of belonging to the classes.
        """
        x = self.convnet(x)
        x = self.densenet(x)
        return x


class OrdinalDenseNet(nn.Module):
    """
    Hidden fully connected parts of the ordinal architecture.
    """
    def __init__(
        self,
        input_size : int,
        hidden_size : int,
        n_classes : int,
        dropout_rate : float
    ):
        """Initializes the fully connected part of an ordinal
        classification neural network.

        Args:
          input_size: the size of the input.
          hidden_size: the size of the hidden layer.
          n_classes: the number of output classes.
          dropout_rate: the dropout rate.
        """
        super(OrdinalDenseNet, self).__init__()

        hidden_size_per_unit = np.round(hidden_size / (n_classes - 1)).astype(int)
        self.dense_hidden = nn.ModuleList(
            [nn.Linear(input_size, hidden_size_per_unit)
             for _ in range(n_classes - 1)]
        )
        self.dense_hidden_dropout = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(n_classes - 1)]
        )
        self.dense_output = nn.ModuleList(
            [nn.Linear(hidden_size_per_unit, 1) for _ in range(n_classes - 1)]
        )

        # Reference vectors for each class, for predictions
        self.target_class = np.ones((n_classes, n_classes - 1),
                                    dtype=np.float32)
        self.target_class[np.triu_indices(n_classes, 0, n_classes - 1)] = 0.0
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x : torch.Tensor) -> List[torch.Tensor]:
        """Performs forward propagation of input data through the
        network.

        Args:
          x: the input data tensor

        Returns:
          A list of the outputs from the dense part of the network. The
          output is interpreted as the distance between consecutive
          class labels.
        """
        xs = [drop(self.leaky_relu(hidden(x)))
              for hidden, drop in zip(self.dense_hidden,
                                      self.dense_hidden_dropout)]
        xs = [torch.sigmoid(output(xc))[:, 0] for output,
              xc in zip(self.dense_output, xs)]
        return xs

    def outputs(self, x : torch.Tensor) -> np.ndarray:
        """Computes network outputs.

        Args:
          x: the input data tensor

        Returns:
          The outputs from the dense part of the network.
        """
        x = self.forward(x)
        return torch.cat(
            [o.unsqueeze(dim=1) for o in x], dim=1
        ).detach().cpu().numpy()

    def predict(self, x : torch.Tensor) -> np.ndarray:
        """Makes a prediction from input data.

        Args:
          x: the input data tensor

        Returns:
          A numpy array with the predicted class labels.
        """
        self.eval()
        outputs = self.outputs(x)
        distances = cdist(outputs, self.target_class, metric='euclidean')
        labels = distances.argmin(axis=1)
        return labels


class OrdinalNet(BrainNet):
    """
    Combination of convolutional and hidden fully connected parts of
    the ordinal architecture (ConvNet + OrdinalDenseNet).
    """
    def __init__(
        self,
        image_shape: Tuple[int, ...],
        n_channels: List[int],
        kernel_size: int,
        stride: int,
        hidden_size: int,
        n_classes: int,
        dropout_rate: float
    ):
        """Initializes an ordinal classification convolutional neural
        network.

        Args:
          image_shape: the shape of the input image.
          n_channels: the number of input channels.
          kernel_size: the size of the kernel.
          stride: the stride of the convolution.
          hidden_size: the size of the hidden layer.
          n_classes: the number of output classes.
          dropout_rate: the dropout rate.
        """
        super(OrdinalNet, self).__init__()
        self.convnet = ConvNet(image_shape, n_channels, kernel_size, stride)
        self.densenet = OrdinalDenseNet(self.convnet.conv_output_size,
                                        hidden_size,
                                        n_classes, dropout_rate)

    def forward(self, x : torch.Tensor) -> List[torch.Tensor]:
        """Performs forward propagation of input data through the
        network.

        Args:
          x: the input data tensor

        Returns:
          A list of the outputs from the dense part of the network. The
          output is interpreted as the distance between consecutive
          class labels.
        """
        x = self.convnet(x)
        x = self.densenet(x)
        return x


class WrapperOrdinalNet(nn.Module):
    """
    Wrapper for the ordinal network to interpret ordinal outputs as
    class membership probabilities.
    """
    def __init__(self, original_net : OrdinalNet):
        """Initializes a wrapper for an ordinal architecture network.

        Args:
          original_net: original ordinal neural network.
        """
        super(WrapperOrdinalNet, self).__init__()
        self.original_net = original_net
        self.target_class = torch.from_numpy(
            self.original_net.densenet.target_class
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """Performs forward propagation of input data through the
        network.

        Args:
          x: the input data tensor

        Returns:
          The output from the dense part of the network. The output is
          interpreted as the distance between the output and the target
          classes, representing the probability of belonging to the
          classes.
        """
        output_list = self.original_net(x)
        output_tensor = torch.stack(output_list, dim=1)
        distances_tensor = -torch.cdist(output_tensor, self.target_class)
        return distances_tensor


    def predict(self, x : torch.Tensor) -> torch.Tensor:
        """Makes a prediction from input data.

        Args:
          x: the input data tensor.

        Returns:
          A tensor with the network predictions.
        """
        self.eval()
        x = self.original_net.convnet(x)
        return self.original_net.densenet.predict(x)


def ordinal_distance_loss(n_classes : int, device : torch.device):
    """Gets a loss function for ordinal classification.

    Args:
      n_classes: the number of output classes.
      device: the computing device.

    Returns:
      The ordinal loss function.
    """
    target_class = np.ones((n_classes, n_classes-1), dtype=np.float32)
    target_class[np.triu_indices(n_classes, 0, n_classes-1)] = 0.0
    target_class = torch.tensor(target_class, device=device)
    mse = nn.MSELoss(reduction='sum')

    def _ordinal_distance_loss(net_output, target):
        net_output = torch.stack(net_output, dim=1)
        target = target_class[target]
        return mse(net_output, target)

    return _ordinal_distance_loss
