#!/usr/bin/env python3

"""
Produces visual representations of ROI analysis from convolutional
neural networks for Parkinson's disease.

Author: Angel Sevilla Molina
source: https://github.com/angelsevillamol/xai-parkinsons-cnn/blob/main/plot_rankings.py
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

from config_reader import ConfigReader
from plot_generator import PlotGenerator


N_TARGETS = 4
METHODS = ['gradcam', 'gbp', 'lrp']
MODELS = ['beta1', 'beta2', 'beta3', 'gamma', 'nominal']
PLOT_TYPES = ['barplots', 'boxplots', 'comparative']


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
      Parsed arguments from the CLI.
    """
    parser = argparse.ArgumentParser(
        description='Produce visual representations of ROI analysis from '
                    'convolutional neural networks for Parkinson\'s disease.'
        )
    parser.add_argument('--method', type=str, choices=METHODS,
                        required=True, help='XAI technique to visualize.')
    parser.add_argument('--model', type=str, choices=MODELS, default=None,
        help='Network model. Not required for comparative barplots.')
    parser.add_argument('--plot_type', type=str, choices=PLOT_TYPES,
                        required=True, help='Type of plot to generate.')

    return parser.parse_args()


def validate_args(method : str, model : str, plot_type : str) -> None:
    """Checks if the command line arguments are valid.

    Args:
      method: indicates the name of the XAI technique to represent.
      model: indicates the network configuration to represent.
      plot_type: indicates the type of diagram to generate.

    Raises:
      ValueError: if any argument is invalid.
    """
    # Validate the explanatory method
    if method not in METHODS:
        raise ValueError(f'The method {method} is not valid. '
                         f'Choose from: {METHODS}')

    # Validate the network model
    if plot_type != 'comparative' and model not in MODELS:
        raise ValueError(f'The model {model} is not valid '
                         f'Choose from: {MODELS}')

    # Validate the graph type
    if plot_type not in PLOT_TYPES:
        raise ValueError(f'The type {plot_type} is not valid. '
                         f'Choose from: {PLOT_TYPES}')

    # Validate that if the plot is comparative,
    # the model is not required
    if plot_type == 'comparative' and model is not None:
        raise ValueError('Comparative plots should not specify a model. '
                         'These are performed on beta3 and nominal models.')


def generate_barplots(
    method : str,
    model : str,
    barplots_path : str,
    grouped_stats_path : str
) -> None:
    """Generates barplot rankings for all classes, ordered by
    relevance for a given XAI method and network model.

    Requires aggregated ROI analysis results.

    Args:
      method: indicates the name of the XAI technique to represent.
      model: indicates the network configuration to represent.
      barplots_path : indicates the path to the bar plots directory.
      grouped_stats_path: indicates the path to the grouped stats
                          directory.
    """
    # Check or create barplots directory
    os.makedirs(barplots_path, exist_ok=True)

    # Read the DataFrame with the results of the analysis
    input_filename = f'{method}_{model}.csv'
    input_path = os.path.join(grouped_stats_path, input_filename)
    df = pd.read_csv(input_path, index_col=0)

    # For each target
    for target in range(N_TARGETS):
        output_filename = f'{method}_{model}_target{target}.png'
        output_path = os.path.join(barplots_path, output_filename)
        try:
            # Get data for the specific target
            df_target = PlotGenerator.get_barplots_data(df, str(target))
            # Creates the bar plot
            fig = PlotGenerator.barplot(df_target, xlabel='Average Intensity',
                    ylabel='Harvard-Oxford Subcortical and Cortical Regions')
            fig.savefig(output_path)
            plt.close(fig)
        except KeyError as error:
            sys.exit(f'Error: {error}')
        except ValueError as error:
            sys.exit(f'Error: {error}')


def generate_boxplots(
    method : str,
    model : str,
    boxplots_path : str,
    splits_stats_path : str
) -> None:
    """Generates boxplot rankings for all classes, ordered by
    relevance for a given XAI method and network model.

    Requires aggregated ROI analysis results.

    Args:
      method: indicates the name of the XAI technique to represent.
      model: indicates the network configuration to represent.
      boxplots_path : indicates the path to the box plots directory.
      splits_stats_path: indicates the path to the stats directory.
    """
    # Check or create boxplots directory
    os.makedirs(boxplots_path, exist_ok=True)

    # Read the DataFrames with the results of the analysis
    dfs = []
    for filename in os.listdir(splits_stats_path):
        if filename.startswith(f'{method}_{model}_'):
            file_path = os.path.join(splits_stats_path, filename)
            df = pd.read_csv(file_path)
            dfs.append(df)

    # For each target
    for target in range(N_TARGETS):
        output_filename = f'{method}_{model}_target{target}.png'
        output_path = os.path.join(boxplots_path, output_filename)
        try:
            # Get data for the specific target
            df_target = PlotGenerator.get_boxplots_data(dfs, target)
            # Creates the box plot
            fig = PlotGenerator.boxplot(df_target, xlabel='Average Intensity',
                    ylabel='Harvard-Oxford Subcortical and Cortical Regions')
            fig.savefig(output_path)
            plt.close(fig)
        except KeyError as error:
            sys.exit(f'Error: {error}')
        except ValueError as error:
            sys.exit(f'Error: {error}')


def generate_comparative_plots(
    method : str,
    comparative_path : str,
    grouped_stats_path : str
) -> None:
    """Generates bar chart rankings, ordered by relevance for all
    classes, comparing the 'beta3' and 'nominal' models.

    Requires aggregated ROI analysis results.

    Args:
      method: indicates the name of the XAI technique to represent.
      comparative_path : indicates the path to the comparative plots
                         directory.
      grouped_stats_path: indicates the path to the grouped stats
                          directory.
    """
    # Check or create comparative plots directory
    os.makedirs(comparative_path, exist_ok=True)

    # Read the DataFrames with the results of the analysis
    beta3_filename = f'{method}_beta3.csv'
    nominal_filename = f'{method}_nominal.csv'
    beta3_path = os.path.join(grouped_stats_path, beta3_filename)
    nominal_path = os.path.join(grouped_stats_path, nominal_filename)
    df_beta3 = pd.read_csv(beta3_path, index_col=0)
    df_nominal = pd.read_csv(nominal_path, index_col=0)

    # For each target
    for target in range(N_TARGETS):
        output_filename = f'{method}_target{target}.png'
        output_path = os.path.join(comparative_path, output_filename)
        try:
            # Get data for the specific target
            df_target = PlotGenerator.get_comparative_data(
                df_nominal, df_beta3, target)
            # Creates the comparative plot
            fig = PlotGenerator.comparative_plot(
                df_target, model1='nominal', model2='beta3',
                xlabel='Average Intensity',
                ylabel='Harvard-Oxford Subcortical and Cortical Regions')
            fig.savefig(output_path)
            plt.close(fig)
        except KeyError as error:
            sys.exit(f'Error: {error}')
        except ValueError as error:
            sys.exit(f'Error: {error}')


def main() -> None:
    """Produces visual representations of ROI analysis from
    convolutional neural networks for Parkinson's disease.
    """

    # Read arguments from the parser
    args = parse_arguments()
    method, model, plot_type = args.method, args.model, args.plot_type

    # Validate the arguments
    try:
        validate_args(method, model, plot_type)
    except ValueError as error:
        sys.exit(f'Error: {error}')

    # Read the paths from the configuration file
    try:
        config_reader = ConfigReader('config.ini')
        barplots_path = config_reader.get_barplots_path()
        boxplots_path = config_reader.get_boxplots_path()
        comparative_path = config_reader.get_comparative_path()
        splits_stats_path = config_reader.get_splits_stats_path()
        grouped_stats_path = config_reader.get_grouped_stats_path()
    except FileNotFoundError as error:
        sys.exit(f'Error: {error}')
    except KeyError as error:
        sys.exit(f'Error: {error}')

    # Generate the corresponding diagram for the method and model
    if plot_type == 'barplots':
        generate_barplots(method, model, barplots_path, grouped_stats_path)
    elif plot_type == 'boxplots':
        generate_boxplots(method, model, boxplots_path, splits_stats_path)
    elif plot_type == 'comparative':
        generate_comparative_plots(method, comparative_path, grouped_stats_path)


if __name__ == '__main__':
    main()
