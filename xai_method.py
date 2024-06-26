#!/usr/bin/env python3

"""
Handles explanatory techniques for 3D CNN models for classification
tasks.

Author: Angel Sevilla Molina
source: https://github.com/angelsevillamol/xai-parkinsons-cnn/blob/main/xai_method.py
"""

import torch
from captum.attr import LayerAttribution, LayerGradCam, GuidedBackprop

from lrp import LRP


class XAIMethod:
    """
    Handles explanatory techniques for 3D CNN models for classification
    tasks.

    Enables the initialization and application of different XAI
    techniques on a given convolutional network.
    """
    def __init__(
        self,
        model : torch.nn.Module,
        method : str,
        layer : torch.nn.modules.conv.Conv3d=None
    ):
        """Initializes an explanatory method instance.

        Args:
          model: the pre-trained model.
          method: indicates the name of the XAI technique.
                  It can be 'gradcam', 'gbp' or 'lrp'.
          layer: convolutional layer on which the explanatory method is
                 to be applied. It is only necessary for the Grad-CAM
                 method.

        Raises:
          ValueError: if the layer is not convolutional or does not
                      belong to the model.
          NotImplementedError: if the XAI method is not implemented.
        """
        self.__method = method
        if layer:
            # If the layer does not belong to the model
            if layer not in list(model.modules()):
                raise ValueError('specified layer does not belong to the model')
            # If the layer is not convolutional
            if not isinstance(layer, torch.nn.modules.conv.Conv3d):
                raise ValueError('specified layer must be a convolutional layer')

        try:
            self.__inject_method(model, layer)
        except NotImplementedError:
            raise NotImplementedError('method not implemented')

    def get_method(self) -> str:
        """Returns the name of the built-in explanatory method.

        Returns:
          A string that indicates the name of the XAI technique.
        """
        return self.__method

    def __inject_method(
        self,
        net : torch.nn.Module,
        layer : torch.nn.modules.conv.Conv3d
    ) -> None:
        """Injects the explanatory method on the neural network to
        generate the attribution maps.

        Args:
          net: the pre-trained model.
          layer: convolutional layer on which the explanatory method is
                 to be applied. It is only necessary for the Grad-CAM
                 method.

        Raises:
          NotImplementedError: if the XAI method is not implemented.
        """
        # Create the instance of the corresponding explanatory method
        if self.__method == 'gradcam':
            self.__method_inst = LayerGradCam(net, layer=layer)
        elif self.__method == 'gbp':
            self.__method_inst = GuidedBackprop(net)
        elif self.__method == 'lrp':
            self.__method_inst = LRP(net)
        else:
            raise NotImplementedError('method not implemented')

    def apply(
        self,
        input_t : torch.Tensor,
        target : int,
        postprocess : bool = False
    ) -> torch.Tensor:
        """Applies the explanatory method on an input image, generating
        a heatmap for the prediction task for a target class.

        Args:
          input_t: the input image tensor.
          target: the target class to predict.
          postprocess: indicates whether to apply postprocessing to the
                       output image.

        Returns:
          The heatmap generated by the method during evaluation.

        Raises:
          ValueError: if the input image tensor has invalid shape.
        """
        # If the input image tensor has invalid shape
        if len(input_t.shape) != 5:
            raise ValueError('invalid input tensor shape')

        input_t.requires_grad = True
        target_t = torch.tensor(target)
        try:
            # In case of Grad-CAM, a ReLU is applied on the output
            if self.__method == 'gradcam':
                heatmap_t = self.__method_inst.attribute(input_t,
                                                       target=target_t,
                                                       relu_attributions=True)
            else:
                heatmap_t = self.__method_inst.attribute(input_t, target=target_t)
        except RuntimeError:
            raise ValueError('invalid input tensor shape')

        # Free tensor memory
        del target_t

        # Apply postprocessing to the heatmap
        if postprocess:
            heatmap_t = self.__postprocess(heatmap_t, input_t)
        heatmap_t = heatmap_t.squeeze()
        return heatmap_t

    def __postprocess(
        self,
        heatmap_t : torch.Tensor,
        input_t : torch.Tensor
    ) -> torch.Tensor:
        """Applies a post-processing on the heatmaps generated by the
        explanatory methods.
        The post-processing consists of a normalization in the range 0
        and 1, and a resizing to have the same size as the input image.

        Args:
          heatmap_t: the heatmap generated by the explanatory method.
          input_t: the input image tensor.

        Returns:
          The normalized heatmap in the range 0 and 1, with the same
          size as the original image.
        """
        # In case of Grad-CAM, the heatmap is upsampled
        if self.__method == 'gradcam':
            heatmap_t = LayerAttribution.interpolate(heatmap_t, input_t.shape[2:],
                                                   'trilinear')

        heatmap_t = heatmap_t.detach()

        # Normalize the heatmap to the range [0,1]
        min_val, max_val = torch.min(heatmap_t), torch.max(heatmap_t)
        if (max_val - min_val) < 1e-10:
            heatmap_t = torch.zeros_like(heatmap_t)
        else:
            heatmap_t = (heatmap_t - min_val) / (max_val - min_val)

        heatmap_t = heatmap_t.squeeze()
        return heatmap_t
