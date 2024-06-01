#!/usr/bin/env python3

"""
Allows the management and processing of ROIs to perform heatmap
relevance analysis.

Author: Angel Sevilla Molina
source: https://github.com/angelsevillamol/xai-parkinsons-cnn/blob/main/rois.py
"""

import re
from typing import Tuple, List
import torch
from nibabel.nifti1 import Nifti1Image
from nilearn.image import math_img
from nilearn import datasets


class ROI:
    """
    Class to represent a region of interest with a name and a mask
    tensor.
    """
    def __init__(self, name: str, mask: torch.Tensor):
        """Initialize the region of interest.

        Args:
          name: indicates the name of the region.
          mask: the corresponding binary map of the region.
        """
        self.name = name
        self.mask = mask


class ROIManager:
    """
    Handles regions of interest for medical analysis.

    Processes brain ROIs from NifTI images for detecting relevant
    regions in Parkinson's classification task.
    """

    def __init__(self, atlases: List[str]):
        """Initializes the ROI manager with the specified
        Harvard-Oxford medical atlases.

        Args:
          atlases: indicates the names of the medical atlases to use.

        Raise:
          ValueError: if the specified atlases are invalid.
        """
        valid_atlases = ['cort-maxprob-thr0-1mm', 'sub-maxprob-thr0-1mm',
                         'cort-maxprob-thr25-1mm', 'sub-maxprob-thr25-1mm',
                         'cort-maxprob-thr50-1mm', 'sub-maxprob-thr50-1mm',
                         'cort-maxprob-thr0-2mm', 'sub-maxprob-thr0-2mm',
                         'cort-maxprob-thr25-2mm', 'sub-maxprob-thr25-2mm',
                         'cort-maxprob-thr50-2mm', 'sub-maxprob-thr50-2mm']
        self.__rois = []
        self.__mask_shape = None

        for atlas in atlases:
            if atlas not in valid_atlases:
                raise ValueError('invalid atlas name')

        for atlas in atlases:
            regions = self.__load_atlas(atlas)
            try:
                self.__process_regions(regions['labels'], regions['maps'])
            except ValueError:
                raise ValueError('mask dimension mismatch')

    def __load_atlas(self, atlas: str) -> dict:
        """Loads the specified Harvard-Oxford medical atlas.

        Args:
          atlas: indicates the name of the medical atlas to use.

        Returns:
          A dictionary that contains the labels and maps of the ROIs from
          the atlas.
        """
        regions = datasets.fetch_atlas_harvard_oxford(atlas)
        return {'labels': regions.labels, 'maps': regions.maps}

    def __get_base_key(self, region_key: str) -> str:
        """Gets the base name of an anatomical region.

        Args:
          region_key: indicates the name of the region.

        Returns:
          The base name of the region key, in lowercase, and
          disregarding hemisphere.
        """
        for prefix in ['left ', 'right ']:
            if prefix in region_key:
                return region_key.replace(prefix, '')
        return region_key

    def __process_regions(self, labels: List[str], maps: Nifti1Image) -> None:
        """Processes the regions from medical atlases, unifying
        symmetric regions, and preparing region keys.

        Args:
          labels: indicates the list of region labels.
          maps: the corresponding maps of the regions.

        Raise:
          ValueError: if the dimensions of the masks of the ROIs are
                      different.
        """
        for roi_name in labels:
            idx = labels.index(roi_name)
            # Convert the region name
            roi_key = re.sub(r'\([^)]*\)', '', roi_name).strip().lower()
            base_key = self.__get_base_key(roi_key)
            # Create a binary mask
            roi_mask = math_img(f'img == {idx}', img=maps)
            mask_t = torch.tensor(roi_mask.get_fdata(), dtype=torch.bool)

            # Check mask dimensions
            if self.__mask_shape is None:
                self.__mask_shape = mask_t.shape
            elif self.__mask_shape != mask_t.shape:
                raise ValueError

            # Check if the ROI already exists
            existing_roi = None
            for roi in self.__rois:
                if roi.name == base_key:
                    existing_roi = roi
                    break

            if existing_roi:
                existing_roi.mask = existing_roi.mask.logical_or(mask_t)
            else:
                self.__rois.append(ROI(base_key, mask_t))

    def get_rois(self) -> List[ROI]:
        """Returns the regions of interest.

        Returns:
          A list that contains the considered regions of
          interest.
        """
        return self.__rois


class ROIAnalyzer:
    """
    Performs a relevance analysis of regions of interest in a heatmap.

    Processes brain regions from a heatmap to determine their relevance
    in the classification task of Parkinson's disease.
    """
    def __init__(self, device: torch.device = None):
        """Initializes the ROI relevance analyzer.

        Args:
          device: the analysis device.
        """
        if device is None:
            if torch.cuda.is_available():
                self.__device = torch.device('cuda')
            else:
                self.__device = torch.device('cpu')
        else:
            self.__device = device

    def analyze(self, heatmap_t: torch.Tensor, rois: List[ROI]) -> dict:
        """Analyzes the relevance values of a set of regions of
        interest for a heatmap.

        Args:
          heatmap_t: the 3D tensor of the heatmap.
          rois: the region names and maps to be analyzed.

        Returns:
          A dictionary with the relevance statistics of all regions.
          For each region, the mean and standard deviation values are
          calculated.

        Raises:
          ValueError: if any mask has incompatible dimensions.
        """
        stats = {}

        for roi in rois:
            if roi.mask.shape != heatmap_t.shape:
                raise ValueError('incompatible dimensions')

            mean, std = self.calculate_roi_stats(heatmap_t, roi)
            stats[f'{roi.name} mean'] = mean
            stats[f'{roi.name} std'] = std

        return stats

    def calculate_roi_stats(
        self,
        heatmap_t: torch.Tensor,
        roi: ROI
    ) -> Tuple[float, float]:
        """Calculates the relevance values of a region for a heatmap.
        Relevance values are defined as the mean intensity and standard
        deviation within the region.

        Args:
          heatmap_t: the 3D tensor of the heatmap.
          roi: the region of interest to be analyzed.

        Returns:
          A tuple with the mean and standard deviation values within
          the region.

        Raises:
          ValueError: if the mask has incompatible dimensions.
        """
        if roi.mask.shape != heatmap_t.shape:
            raise ValueError('incompatible dimensions')

        roi.mask = roi.mask.to(self.__device)
        region_t = heatmap_t[roi.mask > 0]

        if region_t.size(0) > 0:
            mean = round(float(region_t.mean()), 8)
            std = round(float(region_t.std()), 8)
        else:
            mean = 0.0
            std = 0.0

        return mean, std
