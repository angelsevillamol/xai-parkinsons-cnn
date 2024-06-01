#!/usr/bin/env python3

"""
Applies unit tests on the xai_method module.

Author: Angel Sevilla Molina
source: https://github.com/angelsevillamol/xai-parkinsons-cnn/blob/main/tests/test_xai_method.py
"""

import warnings
import pytest
import torch

from xai_method import XAIMethod


@pytest.mark.parametrize('method', ['gradcam', 'gbp', 'lrp'])
def test_xai_method_init(model, method, layer):
    """
    Tests a correct initialization of XAIMethod for the explanatory
    methods GBP, Grad-CAM and LRP.

    Args:
      model: the pre-trained model.
      method: indicates the name of the XAI technique.
      layer: convolutional layer on which the explanatory method is
             to be applied.

    Asserts:
      The method used by XAIMethod corresponds to the one indicated in
      the constructor.
    """
    xai_method = XAIMethod(model, method, layer)
    assert xai_method.get_method() == method


@pytest.mark.parametrize('method', [None, '', 'deeplift'])
def test_invalid_method(model, method):
    """
    Tests the initialization of XAIMethod with an unimplemented method.

    Args:
      model: the pre-trained model.
      method: indicates the name of the XAI technique.

    Asserts:
      XAIMethod raises NotImplementedError if the XAI method is not
      implemented.
    """
    with pytest.raises(NotImplementedError, match='method not implemented'):
        XAIMethod(model, method)


def test_incorrect_layer_type(model):
    """
    Tests the initialization of XAIMethod with a non-convolutional
    layer.

    Args:
      model: the pre-trained model.

    Asserts:
      XAIMethod raises ValueError if the layer is not convolutional.
    """
    layer = model.convnet.batch_norms[0]
    with pytest.raises(ValueError,
                       match='specified layer must be a convolutional layer'):
        XAIMethod(model, 'gradcam', layer)


def test_not_instance_layer(model):
    """
    Tests the initialization of XAIMethod with a layer that does not
    belong to the network.

    Args:
      model: the pre-trained model.

    Asserts:
      XAIMethod raises ValueError if the layer does not belong to the
      model.
    """
    layer = torch.nn.modules.conv.Conv3d(1, 16, 3)
    with pytest.raises(ValueError,
                       match='specified layer does not belong to the model'):
        XAIMethod(model, 'gradcam', layer)


def test_invalid_input_tensor_shape(model):
    """
     Tests the operation of XAIMethod if it receives an input tensor
     with an incorrect shape.

    Args:
      model: the pre-trained model.

    Asserts:
      XAIMethod raises ValueError if the input image tensor has invalid
      shape.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module='captum')
        input_t = torch.rand((1, 1, 90, 100, 90), requires_grad=True)
        xai_method = XAIMethod(model, 'gbp')
        with pytest.raises(ValueError, match='invalid input tensor shape'):
            xai_method.apply(input_t, 0)


@pytest.mark.parametrize('method', ['gbp', 'lrp'])
def test_output_shape(model, method, input_shape):
    """
    Tests the shape of the output generated by XAIMethod after applying
    the GBP and LRP methods.

    Args:
      model: the pre-trained model.
      method: indicates the name of the XAI technique.
      input_shape: the shape of the input image.

    Asserts:
      The heatmap has the same shape as the input tensor (excluding
      Grad-CAM).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module='captum')
        input_t = torch.rand(input_shape, requires_grad=True)
        xai_method = XAIMethod(model, method)
        heatmap_t = xai_method.apply(input_t, 0, False)
        assert input_t.shape[2:] == heatmap_t.shape


def test_output_shape_gradcam(model, layer, input_shape):
    """
    Tests the shape of the output generated by XAIMethod after applying
    the Grad-CAM method without using postprocessing.

    Args:
      model: the pre-trained model.
      layer: convolutional layer on which the explanatory method is
             to be applied.
      input_shape: the shape of the input image.

    Asserts:
      The output for Grad-CAM does not have the same shape as the input
      tensor.
      The output shape matches that of the given convolutional layer.
    """
    input_t = torch.rand(input_shape, requires_grad=True)
    xai_method = XAIMethod(model, 'gradcam', layer)
    heatmap_t = xai_method.apply(input_t, 0)
    assert input_t.shape[2:] != heatmap_t.shape
    assert heatmap_t.shape == torch.Size([8, 11, 8])


@pytest.mark.parametrize('method', ['gradcam', 'gbp', 'lrp'])
def test_postprocess_output_shape(model, method, layer, input_shape):
    """
    Tests that the postprocessed output of XAIMethod has the same shape
    as the input image.

    Args:
      model: the pre-trained model.
      method: indicates the name of the XAI technique.
      layer: convolutional layer on which the explanatory method is
             to be applied.
      input_shape: the shape of the input image.

    Asserts:
      The heatmap has the same shape as the input tensor.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module='captum')
        input_t = torch.rand(input_shape, requires_grad=True)
        xai_method = XAIMethod(model, method, layer)
        heatmap_t = xai_method.apply(input_t, 0, True)
        assert input_t.shape[2:] == heatmap_t.shape


@pytest.mark.parametrize('method', ['gradcam', 'gbp', 'lrp'])
def test_postprocess_normalization(model, method, layer, input_shape):
    """
    Tests that the postprocessing of the output of XAIMethod applies a
    normalization of the values in the range [0,1].

    Args:
      model: the pre-trained model.
      method: indicates the name of the XAI technique.
      layer: convolutional layer on which the explanatory method is
             to be applied.
      input_shape: the shape of the input image.

    Asserts:
      The heatmap is normalized between 0 and 1.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module='captum')
        input_t = torch.rand(input_shape, requires_grad=True)
        xai_method = XAIMethod(model, method, layer)
        heatmap_t = xai_method.apply(input_t, 0, True)
        assert heatmap_t.min() >= 0 and heatmap_t.max() <= 1