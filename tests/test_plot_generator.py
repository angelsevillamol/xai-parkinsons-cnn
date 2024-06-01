#!/usr/bin/env python3

"""
Applies unit tests on the plot_generator module.

Author: Angel Sevilla Molina
source: https://github.com/angelsevillamol/xai-parkinsons-cnn/blob/main/tests/test_plot_generator.py
"""

import pytest
import matplotlib.pyplot as plt

from rois import ROIManager
from plot_generator import PlotGenerator


@pytest.mark.parametrize('target', [0, 1, 2, 3])
def test_get_barplots_data(grouped_df, target):
    """
    Tests that the preparation of the data for the bar plots generate
    DataFrames of correct dimension and format.

    Args:
      grouped_df: contains the aggregated experiment information.
      target: value of the target class to represent.

    Asserts:
      The DataFrame has as many rows as regions and three columns.
      The DataFrame contains a column with the name of the regions.
      The DataFrame contains a column with the mean intensity.
      The DataFrame contains a column with the mean standard deviation.
    """
    df = PlotGenerator.get_barplots_data(grouped_df, str(target))
    assert df.shape == (61, 3)
    assert 'Region' in df.columns
    assert 'Mean' in df.columns
    assert 'Std' in df.columns


@pytest.mark.parametrize('target', [None, '', 'Invalid'])
def test_get_barplots_data_invalid_target(grouped_df, target):
    """
    Tests the behavior of the PlotGenerator class when preparing data
    for a bar plot for an invalid target class.

    Args:
      grouped_df: contains the aggregated experiment information.
      target: value of the target class to represent.

    Asserts:
      PlotGenerator raises KeyError if an invalid target has been
      specified.
    """
    with pytest.raises(KeyError, match='invalid target'):
        PlotGenerator.get_barplots_data(grouped_df, target)


@pytest.mark.parametrize('target', [0, 1, 2, 3])
def test_get_barplots_data_normalization(grouped_df, target):
    """
    Tests that the data for bar plots are normalized between 0 and 1.

    Args:
      grouped_df: contains the aggregated experiment information.
      target: value of the target class to represent.

    Asserts:
      The 'Mean' column is between 0 and 1.
      The 'Std' column is between 0 and 1.
    """
    df = PlotGenerator.get_barplots_data(grouped_df, str(target))
    assert df['Mean'].between(0, 1).all()
    assert df['Std'].between(0, 1).all()


def test_barplot(grouped_df):
    """
    Tests that a bar plot is generated correctly when the input data is
    valid.

    Args:
      grouped_df: contains the aggregated experiment information.

    Asserts:
      The generated image is a matplotlib figure.
      The figure has correct axes and labels.
    """
    df = PlotGenerator.get_barplots_data(grouped_df, '0')
    fig = PlotGenerator.barplot(df, 'X Label', 'Y Label')
    assert isinstance(fig, plt.Figure)
    assert fig.get_axes()
    ax = fig.get_axes()[0]
    assert ax.get_xlabel() == 'X Label'
    assert ax.get_ylabel() == 'Y Label'
    assert ax.get_yticklabels()
    plt.close(fig)


def test_barplot_incorrect_data(grouped_df):
    """
    Tests the behavior of the class PlotGenerator when trying to
    generate a bar plot with data that do not have the required
    structure.

    Args:
      grouped_df: contains the aggregated experiment information.

    Asserts:
      PlotGenerator raises ValueError if the DataFrame is incorrect.
    """
    df = PlotGenerator.get_comparative_data(grouped_df, grouped_df, '0')
    with pytest.raises(ValueError, match='incorrect dataframe'):
        PlotGenerator.barplot(df, 'X Label', 'Y Label')


@pytest.mark.parametrize('target', [0, 1, 2, 3])
def test_get_boxplots_data(dfs, target):
    """
    Tests that the preparation of data for box plots generates
    DataFrames of correct dimension and format.

    Args:
      dfs: contains the aggregated experiment information.
      target: value of the target class to represent.

    Asserts:
      The DataFrame has as many columns as considered regions.
      The columns correspond to the names of the regions.
    """
    df = PlotGenerator.get_boxplots_data(dfs, target)
    roi_manager = ROIManager(['cort-maxprob-thr25-2mm',
                              'sub-maxprob-thr25-2mm'])
    roi_names = [roi.name for roi in roi_manager.get_rois()]
    assert df.shape[1] == len(roi_names)
    assert all(col in df.columns for col in roi_names)


@pytest.mark.parametrize('target', [None, '', 4])
def test_get_boxplots_data_invalid_target(dfs, target):
    """
    Tests the behavior of the PlotGenerator class when preparing data
    for a box plot for an invalid target class.

    Args:
      dfs: contains the aggregated experiment information.
      target: value of the target class to represent.

    Asserts:
      PlotGenerator raises KeyError if an invalid target has been
      specified.
    """
    with pytest.raises(KeyError, match='invalid target'):
        PlotGenerator.get_boxplots_data(dfs, target)


@pytest.mark.parametrize('target', [0, 1, 2, 3])
def test_get_boxplots_data_normalization(dfs, target):
    """
    Tests that data for box plots are normalized between 0 and 1.

    Args:
      dfs: contains the aggregated experiment information.
      target: value of the target class to represent.

    Asserts:
      All values in the DataFrame are between 0 and 1.
    """
    df = PlotGenerator.get_boxplots_data(dfs, target)
    assert (df >= 0).all().all() and (df <= 1).all().all()


def test_boxplot(dfs):
    """
    Tests that a box plot is generated correctly when the input data is
    valid.

    Args:
      dfs: contains the aggregated experiment information.

    Asserts:
      The generated image is a matplotlib figure.
      The figure has correct axes and labels.
    """
    data = PlotGenerator.get_boxplots_data(dfs, 0)
    fig = PlotGenerator.boxplot(data, 'X Label', 'Y Label')
    assert isinstance(fig, plt.Figure)
    assert fig.get_axes()
    ax = fig.get_axes()[0]
    assert ax.get_xlabel() == 'X Label'
    assert ax.get_ylabel() == 'Y Label'
    assert ax.get_yticklabels()
    plt.close(fig)


def test_boxplot_incorrect_data(grouped_df):
    """
    Tests the behavior of the PlotGenerator class when trying to
    generate a box plot with data that does not have the required
    structure.

    Args:
      grouped_df: contains the aggregated experiment information.

    Asserts:
      PlotGenerator raises ValueError if the DataFrame is incorrect.
    """
    df = PlotGenerator.get_barplots_data(grouped_df, '0')
    with pytest.raises(ValueError, match='incorrect dataframe'):
        PlotGenerator.boxplot(df, 'X Label', 'Y Label')


@pytest.mark.parametrize('target', [0, 1, 2, 3])
def test_get_comparative_data(grouped_df, target):
    """
    Tests that the preparation of the data for the comparative plots
    generates DataFrames of correct dimension and format.

    Args:
      grouped_df: contains the aggregated experiment information.
      target: value of the target class to represent.

    Asserts:
      The DataFrame has as many rows as regions and three columns.
      The DataFrame contains a column with the name of the regions.
      The DataFrame contains a column with the mean intensity of the
      first model.
      The DataFrame contains a column with the mean intensity of the
      second model.
    """
    df = PlotGenerator.get_comparative_data(grouped_df, grouped_df,
                                            str(target))
    assert df.shape == (61, 3)
    assert 'Region' in df.columns
    assert 'Mean_model1' in df.columns
    assert 'Mean_model2' in df.columns


@pytest.mark.parametrize('target', [None, '', 'Invalid'])
def test_get_comparative_data_invalid_target(grouped_df, target):
    """
    Tests the behavior of the PlotGenerator class when preparing data
    for a comparative plot for an invalid target class.

    Args:
      grouped_df: contains the aggregated experiment information.
      target: value of the target class to represent.

    Asserts:
      PlotGenerator raises KeyError if an invalid target has been
      specified.
    """
    with pytest.raises(KeyError, match='invalid target'):
        PlotGenerator.get_comparative_data(grouped_df, grouped_df, target)


@pytest.mark.parametrize('target', [0, 1, 2, 3])
def test_get_comparative_data_normalization(grouped_df, target):
    """
    Tests that the data for comparative plots are normalized between 0
    and 1.

    Args:
      grouped_df: contains the aggregated experiment information.
      target: value of the target class to represent.

    Asserts:
      The 'Mean_model1' column is between 0 and 1.
      The 'Mean_model2' column is between 0 and 1.
    """
    df = PlotGenerator.get_comparative_data(grouped_df, grouped_df,
                                            str(target))
    assert df['Mean_model1'].between(0, 1).all()
    assert df['Mean_model2'].between(0, 1).all()


def test_comparative_plot(grouped_df):
    """
    Tests that a comparative plot is generated correctly when the input
    data is valid.

    Args:
      grouped_df: contains the aggregated experiment information.

    Asserts:
      The generated image is a matplotlib figure.
      The figure has correct axes and labels.
    """
    df = PlotGenerator.get_comparative_data(grouped_df, grouped_df, '0')
    fig = PlotGenerator.comparative_plot(df, 'Model 1', 'Model 2',
                                         'X Label', 'Y Label')
    assert isinstance(fig, plt.Figure)
    assert fig.get_axes()
    ax = fig.get_axes()[0]
    assert ax.get_xlabel() == 'X Label'
    assert ax.get_ylabel() == 'Y Label'
    assert ax.get_yticklabels()
    plt.close(fig)


def test_comparative_plot_incorrect_data(grouped_df):
    """
    Tests the behavior of the class PlotGenerator when trying to
    generate a comparative plot with data that do not have the
    required structure.

    Args:
      grouped_df: contains the aggregated experiment information.

    Asserts:
      PlotGenerator raises ValueError if the DataFrame is incorrect.
    """
    df = PlotGenerator.get_barplots_data(grouped_df, '0')
    with pytest.raises(ValueError, match='incorrect dataframe'):
        PlotGenerator.comparative_plot(df, 'Model 1', 'Model 2',
                                       'X Label', 'Y Label')
