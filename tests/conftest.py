#!/usr/bin/env python3

"""
Pytest configuration file for setting up test fixtures.

Author: Angel Sevilla Molina
source: https://github.com/angelsevillamol/xai-parkinsons-cnn/blob/main/tests/conftest.py
"""

import os
import pytest
import pandas as pd
import torch
import numpy as np

from network import NominalNet, OrdinalNet
from rois import ROI, ROIAnalyzer


@pytest.fixture(autouse=True)
def clear_cuda_cache():
    """
    Fixture to clear the CUDA cache.
    """
    yield
    torch.cuda.empty_cache()


@pytest.fixture
def device():
    """
    Fixture for the computing device.

    Returns:
      The computing device.
    """
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def input_shape():
    """
    Fixture for the shape of the input image.

    Returns:
      The shape of the input image.
    """
    return torch.Size([1, 1, 91, 109, 91])


@pytest.fixture
def nominal_network_params():
    """
    Fixture for the parameters to initialize the nominal network.

    Returns:
      The parameters to initialize the nominal network.
    """
    return {
        'image_shape': (91, 109, 91),
        'n_channels': [1, 20, 30, 40, 50],
        'kernel_size': 5,
        'stride': 2,
        'hidden_size': 4096,
        'n_classes': 4,
        'dropout_rate': 0.3,
    }


@pytest.fixture
def ordinal_network_params():
    """
    Fixture for the parameters to initialize the ordinal network.

    Returns:
      The parameters to initialize the ordinal network.
    """
    return {
        'image_shape': (91, 109, 91),
        'n_channels': [1, 20, 30, 40, 50],
        'kernel_size': 3,
        'stride': 2,
        'hidden_size': 4096,
        'n_classes': 4,
        'dropout_rate': 0.3,
    }


@pytest.fixture
def nominal_net(nominal_network_params):
    """
    Fixture for the nominal architecture.

    Args:
      nominal_network_params: the parameters to initialize the nominal
                              network.

    Returns:
      An instance of the nominal network
    """
    return NominalNet(**nominal_network_params)


@pytest.fixture
def ordinal_net(ordinal_network_params):
    """
    Fixture for the ordinal architecture.

    Args:
      ordinal_network_params: the parameters to initialize the ordinal
                              network.

    Returns:
      An instance of the ordinal network
    """
    return OrdinalNet(**ordinal_network_params)


@pytest.fixture
def nominal_trained_state_path():
    """
    Fixture for the file path to the trained state of the nominal
    network.

    Returns:
      The path to the trained state file.
    """
    return os.path.abspath('./data/nominal_trained_state.pt')


@pytest.fixture
def ordinal_trained_state_path():
    """
    Fixture for the file path to the trained state of the ordinal
    network.

    Returns:
      The path to the trained state file.
    """
    return os.path.abspath('./data/ordinal_trained_state.pt')


@pytest.fixture
def model(nominal_network_params, nominal_trained_state_path, device):
    """
    Fixture to initialize and return a nominal network with pre-trained
    weights.

    Args:
      nominal_network_params: the parameters to initialize the nominal
                              network.
      nominal_trained_state_path: path to the file with the pre-trained
                                  model.
      device: the computing device.

    Returns:
      An instance of the NominalNet class with loaded weights.
    """
    net = NominalNet(**nominal_network_params)
    weights = torch.load(nominal_trained_state_path, map_location=device)
    net.load_state_dict(weights)
    return net


@pytest.fixture
def layer(model):
    """
    Fixture to return the second last convolutional layer of the
    network model.

    Args:
      model: the pre-trained model.

    Returns:
      The second last convolutional layer.
    """
    net = model
    return net.convnet.convs[-2]


@pytest.fixture
def heatmap_t():
    """
    Fixture to return a random tensor simulating a heatmap.

    Returns:
      The 3D tensor of the heatmap.
    """
    return torch.rand(91, 109, 91)


@pytest.fixture
def roi():
    """
    Fixture to return a ROI with a random binary mask.

    Returns:
      An instance of the ROI class.
    """
    mask_t = torch.tensor(np.random.rand(91, 109, 91) > 0.5, dtype=torch.bool)
    roi = ROI("region name", mask_t)
    return roi


@pytest.fixture
def roi_analyzer(device):
    """
    Fixture to get a ROI analyzer.

    Args:
      device: the computing device.

    Returns:
      An instance of the ROIAnalyzer class.
    """
    return ROIAnalyzer(device)


@pytest.fixture
def grouped_df():
    """
    Fixture to return a DataFrame containing grouped experimental data.

    Returns:
      A DataFrame read from a csv file.
    """
    df = pd.read_csv('./data/gradcam_nominal.csv', index_col=0)
    return df


@pytest.fixture
def dfs():
    """
    Fixture to return a list of DataFrames containing experimental
    data.

    Returns:
      A dictionary of DataFrames read from csv files.
    """
    df1 = pd.read_csv('./data/gradcam_beta3_fold0_split0.csv')
    df2 = pd.read_csv('./data/gradcam_beta3_fold0_split1.csv')
    dfs = [df1, df2]
    return dfs
