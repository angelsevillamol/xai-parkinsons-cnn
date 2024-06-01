#!/usr/bin/env python3

"""
Reads the application's configuration file.

Author: Angel Sevilla Molina
source: https://github.com/angelsevillamol/xai-parkinsons-cnn/blob/main/config_reader.py
"""

import os
import configparser


class ConfigReader:
    """
    Reads the application's configuration file.

    Allows reading the configuration file containing the paths to the
    different files and directories of the application, acting as an
    interface.
    """

    def __init__(self, config_file : str):
        """Initializes the ConfigReader by loading paths from a
        configuration file.

        Args:
          config_file: indicates the name to the configuration file.

        Raises:
          FileNotFoundError: if the configuration file does not exist.
          KeyError: if a key does not exist in the configuration file.
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f'missing configuration file {config_file}')

        config = configparser.ConfigParser()
        config.read(config_file)

        # Load paths from the configuration file
        try:
            self.__data_root_path = os.path.expanduser(config.get(
                'Paths', 'data_root_path'))
            self.__labels_path = os.path.expanduser(config.get(
                'Paths', 'labels_path'))
            self.__project_root_path = os.path.expanduser(config.get(
                'Paths', 'project_root_path'))
            self.__xai_root_path = os.path.expanduser(config.get(
                'Paths', 'xai_root_path'))
            self.__heatmaps_path = os.path.expanduser(config.get(
                'Paths', 'heatmaps_path'))
            self.__stats_path = os.path.expanduser(config.get(
                'Paths', 'stats_path'))
            self.__splits_stats_path = os.path.expanduser(config.get(
                'Paths', 'splits_stats_path'))
            self.__grouped_stats_path = os.path.expanduser(config.get(
                'Paths', 'grouped_stats_path'))
            self.__plots_path = os.path.expanduser(config.get(
                'Paths', 'plots_path'))
            self.__barplots_path = os.path.expanduser(config.get(
                'Paths', 'barplots_path'))
            self.__boxplots_path = os.path.expanduser(config.get(
                'Paths', 'boxplots_path'))
            self.__comparative_path = os.path.expanduser(config.get(
                'Paths', 'comparative_path'))
            self.__planes_path = os.path.expanduser(config.get(
                'Paths', 'planes_path'))
        except configparser.NoOptionError as error:
            raise KeyError(f'missing key: {error}')

    def get_data_root_path(self) -> str:
        """
        Returns the path to the test images directory.

        Returns:
          The path to the test images directory.
        """
        return self.__data_root_path

    def get_labels_path(self) -> str:
        """
        Returns the path to the file containing class labels.

        Returns:
          The path to the file containing class labels.
        """
        return self.__labels_path

    def get_project_root_path(self) -> str:
        """
        Returns the path to the project's root directory.

        Returns:
          The path to the project's root directory.
        """
        return self.__project_root_path

    def get_xai_root_path(self) -> str:
        """
        Returns the path to the output directory.

        Returns:
          The path to the output directory.
        """
        return self.__xai_root_path

    def get_heatmaps_path(self) -> str:
        """
        Returns the path to the directory where heatmaps are stored.

        Returns:
          The path to the directory where heatmaps are stored.
        """
        return self.__heatmaps_path

    def get_stats_path(self) -> str:
        """
        Returns the path to the root directory of relevance analysis.

        Returns:
          The path to the root directory of relevance analysis.
        """
        return self.__stats_path

    def get_splits_stats_path(self) -> str:
        """
        Returns the path to the directory where individual relevance
        analyses are stored.

        Returns:
          The path to the directory where individual relevance analyses
          are stored.
        """
        return self.__splits_stats_path

    def get_grouped_stats_path(self) -> str:
        """
        Returns the path to the directory where grouped relevance analyses
        are stored.

        Returns:
          The path to the directory where grouped relevance analyses
          are stored.
        """
        return self.__grouped_stats_path

    def get_plots_path(self) -> str:
        """
        Returns the path to the root directory where graphs are stored.

        Returns:
          The path to the root directory where graphs are stored.
        """
        return self.__plots_path

    def get_barplots_path(self) -> str:
        """
        Returns the path to the directory where bar graphs are stored.

        Returns:
          The path to the directory where bar graphs are stored.
        """
        return self.__barplots_path

    def get_boxplots_path(self) -> str:
        """
        Returns the path to the directory where boxplot graphs are stored.

        Returns:
          The path to the directory where boxplot graphs are stored.
        """
        return self.__boxplots_path

    def get_comparative_path(self) -> str:
        """
        Returns the path to the directory where comparative bar graphs are
        stored.

        Returns:
          The path to the directory where comparative bar graphs are
          stored.
        """
        return self.__comparative_path

    def get_planes_path(self) -> str:
        """
        Returns the path to the directory where anatomical plane
        visualizations are stored.

        Returns:
          The path to the directory where anatomical plane
          visualizations are stored.
        """
        return self.__planes_path
