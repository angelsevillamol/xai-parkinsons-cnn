#!/usr/bin/env python3

"""
Applies unit tests on the rois module.

Author: Angel Sevilla Molina
source: https://github.com/angelsevillamol/xai-parkinsons-cnn/blob/main/tests/test_rois.py
"""

import pytest
import numpy as np
import torch

from rois import ROI, ROIManager


@pytest.mark.parametrize('atlas', [
    'cort-maxprob-thr25-2mm',
    'sub-maxprob-thr25-2mm'
])
def test_roi_manager_init(atlas):
    """
    Tests a correct initialization of the ROIManager class using a
    Harvard-Oxford medical atlas.

    Args:
      atlas: indicates the name of the medical atlas to use.

    Asserts:
      The number of loaded ROIs should be greater than 0.
    """
    roi_manager = ROIManager([atlas])
    rois = roi_manager.get_rois()
    assert len(rois) > 0


def test_roi_manager_invalid_atlas():
    """
    Tests the initialization of ROIManager with an invalid atlas.

    Asserts:
      ROIManager raises ValueError if the specified atlas is invalid.
    """
    with pytest.raises(ValueError, match='invalid atlas name'):
        ROIManager(['invalid-atlas'])


def test_roi_manager_multiple_atlas_init():
    """
    Tests the initialization of ROIManager with multiple medical
    atlases, specifically cortical and subcortical regions.

    Asserts:
      'insular cortex' should be part of the ROIs.
      'caudate' should be part of the ROIs.
    """
    roi_manager = ROIManager(['cort-maxprob-thr25-2mm',
                              'sub-maxprob-thr25-2mm'])
    rois = roi_manager.get_rois()
    assert 'insular cortex' in [roi.name for roi in rois]
    assert 'caudate' in [roi.name for roi in rois]


@pytest.mark.parametrize('atlas', [
    'cort-maxprob-thr25-2mm',
    'sub-maxprob-thr25-2mm'
])
def test_roi_manager_names_lower_case(atlas):
    """
    Tests that the region names returned by ROIManager are in lowercase.

    Args:
      atlas: indicates the name of the medical atlas to use.

    Asserts:
      The names of all ROIs should be in lowercase.
    """
    roi_manager = ROIManager([atlas])
    rois = roi_manager.get_rois()
    for roi in rois:
        assert roi.name.islower()


@pytest.mark.parametrize('left_name, right_name, name', [
    ('right putamen', 'left putamen', 'putamen'),
    ('right accumbens', 'left accumbens', 'accumbens'),
    ('right amygdala', 'left amygdala', 'amygdala'),
    ('right caudate', 'left caudate', 'caudate')
    ])
def test_roi_manager_example_regions_unification(left_name, right_name, name):
    """
    Tests that the region names in the atlases loaded by ROIManager are
    unified.

    Args:
      left_name: indicates the name of the left hemisphere region.
      right_name: indicates the name of the right hemisphere region.
      name: indicates the unified name of the region.

    Asserts:
      The unified name should be in the ROIs.
      The left hemisphere name should not be in the ROIs.
      The right hemisphere name should not be in the ROIs.
    """
    roi_manager = ROIManager(['sub-maxprob-thr50-2mm'])
    rois = roi_manager.get_rois()
    assert name in [roi.name for roi in rois]
    assert left_name not in [roi.name for roi in rois]
    assert right_name not in [roi.name for roi in rois]


def test_roi_manager_regions_unification():
    """
    Tests that there are no independent regions of the left and right
    hemispheres in the regions loaded by ROIManager.

    Asserts:
      No region should start with 'left ' or 'right '.
    """
    roi_manager = ROIManager(['sub-maxprob-thr50-2mm'])
    rois = roi_manager.get_rois()
    for roi in rois:
        assert not roi.name.startswith('left ')
        assert not roi.name.startswith('right ')


def test_roi_manager_masks_binary():
    """
    Tests that the masks of the ROIs returned by the ROIManager are
    binary.

    Asserts:
      The values of the masks of the ROIs should be only 0 or 1.
    """
    roi_manager = ROIManager(['cort-maxprob-thr25-2mm',
                              'sub-maxprob-thr25-2mm'])
    rois = roi_manager.get_rois()
    for roi in rois:
        unique_values = torch.unique(roi.mask).numpy()
        assert np.array_equal(unique_values, [0, 1])


def test_roi_manager_incorrect_dimension():
    """
    Tests the initialization of ROIManager with medical atlases of
    different dimensions.

    Asserts:
      ROIManager raises ValueError if the mask dimensions are
      incompatible.
    """
    with pytest.raises(ValueError, match='mask dimension mismatch'):
        ROIManager(['cort-maxprob-thr25-2mm', 'sub-maxprob-thr25-1mm'])


def test_roi_analyzer_incompatible_mask(roi_analyzer, heatmap_t):
    """
    Tests the analysis of a region in a heatmap with a mask of different
    dimensions.

    Args:
      roi_analyzer: the ROI analyzer.
      heatmap_t: the 3D tensor of the heatmap.

    Asserts:
      ROIAnalyzer raises ValueError if the mask has incompatible
      dimensions.
    """
    mask_t = torch.tensor(np.random.rand(91, 110, 91) > 0.5, dtype=torch.bool)
    roi = ROI('incompatible region', mask_t)
    with pytest.raises(ValueError, match='incompatible dimensions'):
        roi_analyzer.calculate_roi_stats(heatmap_t, roi)


def test_roi_analyzer_null_mask(roi_analyzer, heatmap_t):
    """
    Tests the analysis of a region in a heatmap with a mask whose values
    are 0.

    Args:
      roi_analyzer: the ROI analyzer.
      heatmap_t: the 3D tensor of the heatmap.

    Asserts:
      The mean intensity should be 0.
      The standard deviation should be 0.
    """
    roi = ROI('region name', torch.zeros(91, 109, 91, dtype=torch.bool))
    mean, std = roi_analyzer.calculate_roi_stats(heatmap_t, roi)
    assert heatmap_t.sum() != 0.0
    assert mean == 0.0 and std == 0.0


def test_roi_analyzer_null_image(roi_analyzer, roi):
    """
    Tests the analysis of a region in a heatmap where all values are 0.

    Args:
      roi_analyzer: the ROI analyzer.
      roi: the region of interest to be analyzed.

    Asserts:
      The mean intensity should be 0.
      The standard deviation should be 0.
    """
    heatmap_t = torch.zeros(91, 109, 91)
    mean, std = roi_analyzer.calculate_roi_stats(heatmap_t, roi)
    assert roi.mask.sum() != 0.0
    assert mean == 0.0 and std == 0.0


def test_roi_analyzer_positive_image(roi_analyzer, roi):
    """
    Tests the analysis of a region in a heatmap where all values are 1.

    Args:
      roi_analyzer: the ROI analyzer.
      roi: the region of interest to be analyzed.

    Asserts:
      The mean intensity should be 1.
      The standard deviation should be 0.
    """
    heatmap_t = torch.ones(91, 109, 91)
    mean, std = roi_analyzer.calculate_roi_stats(heatmap_t, roi)
    assert mean == 1.0 and std == 0.0


def test_calculate_roi_stats(roi_analyzer, heatmap_t, roi):
    """
    Tests the analysis of a region in a randomly generated heatmap.

    Args:
      roi_analyzer: the ROI analyzer.
      heatmap_t: the 3D tensor of the heatmap.
      roi: the region of interest to be analyzed.

    Asserts:
      The mean intensity should be between 0 and 1.
      The standard deviation should be between 0 and 1.
    """
    mean, std = roi_analyzer.calculate_roi_stats(heatmap_t, roi)
    assert 0.0 <= mean and mean <= 1.0
    assert 0.0 <= std and std <= 1.0


def test_analyze_multiple_rois(roi_analyzer, heatmap_t):
    """
    Tests the analysis of a set of ROIs in a heatmap and check that it
    returns an appropriate number of statistics.

    Args:
      roi_analyzer: the ROI analyzer.
      heatmap_t: the 3D tensor of the heatmap.

    Asserts:
      The number of generated statistics should be twice the number of
      ROIs (mean and standard deviation are calculated per region).
    """
    roi_manager = ROIManager(['cort-maxprob-thr25-2mm'])
    rois = roi_manager.get_rois()
    stats = roi_analyzer.analyze(heatmap_t, rois)
    assert len(stats) == 2 * len(rois)


def test_analyze_stat_names(roi_analyzer, heatmap_t, roi):
    """
    Tests the analysis of a region on a heatmap and check that the
    statistics are correctly named.

    Args:
      roi_analyzer: the ROI analyzer.
      heatmap_t: the 3D tensor of the heatmap.
      roi: the region of interest to be analyzed.

    Asserts:
      The number of calculated statistics per region should be two.
      The mean intensity in the region should be in the calculated
      statistics.
      The standard deviation of intensity in the region should be in
      the calculated statistics.
    """
    stats = roi_analyzer.analyze(heatmap_t, [roi])
    assert len(stats) == 2
    assert f'{roi.name} mean' in stats
    assert f'{roi.name} std' in stats


def test_analyze_empty_list(roi_analyzer, heatmap_t):
    """
    Tests the analysis of a heatmap with an empty list of ROIs.

    Args:
      roi_analyzer: the ROI analyzer.
      heatmap_t: the 3D tensor of the heatmap.

    Asserts:
      No statistics should be calculated.
    """
    results = roi_analyzer.analyze(heatmap_t, [])
    assert results == {}
