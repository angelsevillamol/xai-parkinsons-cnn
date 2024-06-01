#!/usr/bin/env python3

"""
Applies unit tests on the network module.

Author: Angel Sevilla Molina
source: https://github.com/angelsevillamol/xai-parkinsons-cnn/blob/main/tests/test_network.py
"""

import pytest
import numpy as np
import torch

import network


@pytest.mark.parametrize('kernel_size, stride', [(3, 2), (5, 2)])
def test_conv_output_shape(input_shape, kernel_size, stride):
    """
    Tests the expected output shape for a convolutional layer returned
    by the function conv_output_shape.

    Args:
      input_shape: the shape of the input image.
      kernel_size: the size of the kernel.
      stride: the stride of the convolution.

    Asserts:
      The calculated shape corresponds to the expected shape.
    """
    shape = np.array(input_shape[2:])
    result = network.conv_output_shape(shape, kernel_size, stride)
    expected = np.floor(((shape - (kernel_size-1) - 1) / stride) + 1)
    assert np.all(result == expected)


def test_nominal_net_init(nominal_network_params):
    """
    Tests a correct initialization of NominalNet from valid input
    parameters.

    Args:
      nominal_network_params: the parameters to initialize the nominal
                              network.

    Asserts:
      The returned network is an instance of NominalNet.
      The returned network has the parameters specified by the user.
    """
    net = network.NominalNet(**nominal_network_params)
    assert isinstance(net, network.NominalNet)
    assert net.convnet.n_channels == nominal_network_params['n_channels']

    # Calculate the output shape after convolutions
    shape = np.array(nominal_network_params['image_shape'])
    for _ in range(len(nominal_network_params['n_channels']) - 1):
        shape = network.conv_output_shape(
            shape,
            nominal_network_params['kernel_size'],
            nominal_network_params['stride']
        )

    conv_output_shape = (nominal_network_params['n_channels'][-1] *
                         np.prod(shape).astype(int))
    assert net.convnet.conv_output_size == conv_output_shape
    assert net.densenet.dense_hidden.in_features == conv_output_shape

    # Check that the network stores the rest of the hyperparameters
    assert net.densenet.dense_hidden.out_features == (
        nominal_network_params['hidden_size'])
    assert net.densenet.dense_output.out_features == (
        nominal_network_params['n_classes'])
    assert net.densenet.dense_hidden_dropout.p == (
        nominal_network_params['dropout_rate'])


def test_ordinal_net_init(ordinal_network_params):
    """
    Tests a correct initialization of OrdinalNet from valid input
    parameters.

    Args:
      ordinal_network_params: the parameters to initialize the ordinal
                              network.

    Asserts:
      The returned network is an instance of OrdinalNet.
      The returned network has the parameters specified by the user.
    """
    net = network.OrdinalNet(**ordinal_network_params)
    assert isinstance(net, network.OrdinalNet)
    assert net.convnet.n_channels == ordinal_network_params['n_channels']

    # Calculate the output shape after convolutions
    shape = np.array(ordinal_network_params['image_shape'])
    for _ in range(len(ordinal_network_params['n_channels']) - 1):
        shape = network.conv_output_shape(
            shape,
            ordinal_network_params['kernel_size'],
            ordinal_network_params['stride']
        )

    conv_output_shape = (ordinal_network_params['n_channels'][-1] *
                         np.prod(shape).astype(int))
    assert net.convnet.conv_output_size == conv_output_shape
    assert net.densenet.dense_hidden[0].in_features == conv_output_shape

    # Check that the network stores the rest of the hyperparameters
    assert all(dd.p == ordinal_network_params['dropout_rate']
               for dd in net.densenet.dense_hidden_dropout)


def test_wrapper_ordinal_net_init(ordinal_network_params):
    """
    Tests a correct initialization of WrapperOrdinalNet from valid
    input parameters.

    Args:
      ordinal_network_params: the parameters to initialize the ordinal
                              network.

    Asserts:
      The returned network is an instance of WrapperOrdinalNet.
      The wrapped network is the original ordinal net.
    """
    ordinal_net = network.OrdinalNet(**ordinal_network_params)
    wrapper_net = network.WrapperOrdinalNet(ordinal_net)
    assert isinstance(wrapper_net, network.WrapperOrdinalNet)
    assert wrapper_net.original_net is ordinal_net


def test_nominal_net_forward(nominal_network_params, input_shape):
    """
    Tests that the class NominalNet returns a consistent output for a
    random input data.

    Args:
      nominal_network_params: the parameters to initialize the nominal
                              network.
      input_shape: the shape of the input image.

    Asserts:
      The output batch must be the same as the input batch.
      The output shape must match the number of classes.
    """
    net = network.NominalNet(**nominal_network_params)
    input_t = torch.rand(input_shape)
    output_t = net(input_t)
    assert output_t.shape[0] == input_shape[0]
    assert output_t.shape[1] == nominal_network_params['n_classes']


def test_ordinal_net_forward(ordinal_network_params, input_shape):
    """
    Tests that the class OrdinalNet returns a consistent output for a
    random input data.

    Args:
      ordinal_network_params: the parameters to initialize the ordinal
                              network.
      input_shape: the shape of the input image.

    Asserts:
      The size of the output must match the number of class intervals.
    """
    net = network.OrdinalNet(**ordinal_network_params)
    input_t = torch.rand(input_shape)
    output_t = net(input_t)
    assert len(output_t) == ordinal_network_params['n_classes'] - 1


def test_wrapper_ordinal_net_forward(ordinal_network_params, input_shape):
    """
    Tests that the class WrapperOrdinalNet returns a consistent output
    for a random input data.

    Args:
      ordinal_network_params: the parameters to initialize the ordinal
                              network.
      input_shape: the shape of the input image.

    Asserts:
      The output batch must be the same as the input batch.
      The output shape must match the number of classes.
    """
    ordinal_net = network.OrdinalNet(**ordinal_network_params)
    wrapper_net = network.WrapperOrdinalNet(ordinal_net)
    input_t = torch.rand(input_shape)
    output_t = wrapper_net(input_t)
    assert output_t.shape[0] == input_shape[0]
    assert output_t.shape[1] == ordinal_network_params['n_classes']


def test_wrapper_ordinal_net_predict(ordinal_network_params, input_shape):
    """
    Tests that the prediction of the class WrapperOrdinalNet matches the
    prediction of the class OrdinalNet for the same input.

    Args:
      ordinal_network_params: the parameters to initialize the ordinal
                              network.
      input_shape: the shape of the input image.

    Asserts:
      The predicted class matches for both networks.
    """
    ordinal_net = network.OrdinalNet(**ordinal_network_params)
    wrapper_net = network.WrapperOrdinalNet(ordinal_net)
    input_t = torch.rand(input_shape)
    ordinal_predicted = ordinal_net.predict(input_t)
    wrapper_predicted = wrapper_net.predict(input_t)
    assert ordinal_predicted == wrapper_predicted


def test_nominal_load_state(
        nominal_network_params,
        nominal_trained_state_path,
        device
):
    """
    Tests that the class NominalNet allows loading a set of pretrained
    weights.

    Args:
      nominal_network_params: the parameters to initialize the nominal
                              network.
      nominal_trained_state_path: path to the file with the pre-trained
                                  model.
      device: the computing device.

    Asserts:
      The weights of the network match those of the pre-trained model.
    """
    net = network.NominalNet(**nominal_network_params).to(device)
    weights = torch.load(nominal_trained_state_path, map_location=device)
    net.load_state_dict(weights)
    for layer, net_weights in net.state_dict().items():
        assert torch.equal(net_weights, weights[layer])


def test_ordinal_load_state(
        ordinal_network_params,
        ordinal_trained_state_path,
        device
):
    """
    Tests that the class OrdinalNet allows loading a set of pretrained
    weights.

    Args:
      ordinal_network_params: the parameters to initialize the ordinal
                              network.
      ordinal_trained_state_path: path to the file with the pre-trained
                                  model.
      device: the computing device.

    Asserts:
      The weights of the network match those of the pre-trained model.
    """
    net = network.OrdinalNet(**ordinal_network_params).to(device)
    weights = torch.load(ordinal_trained_state_path, map_location=device)
    net.load_state_dict(weights)
    for layer, net_weights in net.state_dict().items():
        assert torch.equal(net_weights, weights[layer])
