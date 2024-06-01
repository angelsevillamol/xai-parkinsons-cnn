#!/usr/bin/env python3

"""
Generates graphical representations of ROI relevance analysis from CSV
files.

Author: Angel Sevilla Molina
source: https://github.com/angelsevillamol/xai-parkinsons-cnn/blob/main/plot_generator.py
"""

from typing import List
import matplotlib.pyplot as plt
import pandas as pd


class PlotGenerator:
    """
    Generates graphical representations of ROI relevance analysis.

    Processes ROI relevance analysis information from CSV files and
    generates different types of visual representations, such as bar
    charts, box plots, and comparative rankings.
    """
    @staticmethod
    def get_barplots_data(df : pd.DataFrame, target : str) -> pd.DataFrame:
        """Reads data from a CSV file and prepares it for ranking bar
        representation.

        Args:
          df: contains the aggregated experiment information.
          target: value of the target class to represent.

        Returns:
          A DataFrame with the necessary information to create a bar
          chart representing the results obtained for a method, model,
          and target class.

          The DataFrame has three columns:
          - 'Region' indicates the name of the brain region.
          - 'Mean' indicates the average intensity value.
          - 'Std' indicates the standard deviation of intensity.

        Raises:
          KeyError: if an invalid target has been specified.
        """
        # Check if the DataFrame contains the given target
        if target not in df.columns:
            raise KeyError('invalid target')

        # Get the names of the regions
        regions = set()
        for row in df.index:
            region = row.replace(' mean', '').replace(' std', '')
            regions.add(region)

        # Create a new DataFrame with the necessary data
        data = {'Region': [], 'Mean': [], 'Std': []}
        for region in regions:
            mean_row = f'{region} mean'
            std_row = f'{region} std'
            data['Region'].append(region)
            data['Mean'].append(df.loc[mean_row, target])
            data['Std'].append(df.loc[std_row, target])

        new_df = pd.DataFrame(data)
        new_df = new_df.sort_values(by='Mean', ascending=True)
        return new_df

    @staticmethod
    def barplot(
        df : pd.DataFrame,
        xlabel : str='',
        ylabel : str=''
    ) -> plt.Figure:
        """Creates a bar plot showing the relevance of the ROIs for a
        method, model, and class.

        Requires aggregated ROI analysis results.

        Args:
          df: a DataFrame with the data needed to make the bar plot.
              Read PlotGenerator.get_barplots_data() for more
              information.
          xlabel: indicates the label for the x-axis.
          ylabel: indicates the label for the y-label.

        Returns:
          A figure representing a bar chart of the given data.

        Raises:
          ValueError: if the DataFrame is incorrect.
        """
        # Validate the DataFrame format
        columns = {'Region', 'Mean', 'Std'}
        if not columns.issubset(df.columns):
            raise ValueError('incorrect dataframe')

        fig, ax = plt.subplots(figsize=(14, 13))
        bars = ax.barh(df['Region'], df['Mean'], color='red')

        # Set plot limits and labels
        expand = df['Mean'].max() - df['Mean'].min()
        ax.set_xlim(df['Mean'].min() - expand * 0.1,
                    df['Mean'].max() + expand * 0.2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Add to each bar its mean and standard deviation value
        for i, (bar, mean, std) in enumerate(zip(bars, df['Mean'], df['Std'])):
            plt.text(bar.get_width() + expand * 0.03, i,
                     f'{mean:.4f} ({std:.2f})', va='center', ha='left',
                     color='black', fontsize=10)

        plt.tight_layout()
        return fig

    @staticmethod
    def get_boxplots_data(
        dfs : List[pd.DataFrame],
        target : int
    ) -> pd.DataFrame:
        """Reads data from CSV files for all experiments and prepares
        it for box plot representation.

        Args:
          dfs: contains the aggregated experiment information.
          target: value of the target class to represent.

        Returns:
          A DataFrame with the necessary information to create a
          box plot representing the results obtained for a method,
          model, and target class.

          The DataFrame has as many columns as ROIs to consider.
          Each row corresponds to a test NifTI image, and the values in
          the cells indicate the relevance assigned to that image in
          that region.

        Raises:
          KeyError: if an invalid target has been specified.
        """
        combined_data = []

        # Read all experiment data files
        for df in dfs:
            # Check if the DataFrame contains the given target
            if target not in df['target'].values:
                raise KeyError('invalid target')

            # Select data for the target class
            df_targeted = df[df['target'] == target]

            # Get mean values
            mean_cols = [c for c in df.columns if c.endswith(' mean')]
            df_filtered = df_targeted[mean_cols]
            df_filtered.columns = [c.replace(' mean', '')
                                   for c in df_filtered.columns]

            # Add values to the dataset
            combined_data.append(df_filtered)

        # Unify all data in a single dataframe
        if combined_data:
            combined_df = pd.concat(combined_data, axis=0)
        else:
            combined_df = pd.DataFrame()
        return combined_df

    @staticmethod
    def boxplot(
        df : pd.DataFrame,
        xlabel : str='',
        ylabel : str=''
    ) -> plt.Figure:
        """Creates a box plot showing the relevance of the ROIs for a
        method, model, and class.

        Args:
          df: a DataFrame with the data needed to make the box plot.
              Read PlotGenerator.get_boxplot_data() for more
              information.
          xlabel: indicates the label for the x-axis.
          ylabel: indicates the label for the y-axis.

        Returns:
          A figure representing a box plot of the given data.

        Raises:
          ValueError: if the DataFrame is incorrect.
        """
        # Validate the DataFrame format
        if df.empty or df.shape[1] < 1:
            raise ValueError('incorrect dataframe')

        # Check the type of the DataFrame elements
        for i in range(df.shape[1]):
            for value in df.iloc[1:, i]:
                if not isinstance(value, float) and not pd.isna(value):
                    raise ValueError('incorrect dataframe')

        # Calculate and sort the average intensity values
        avg_intensities = df.mean().sort_values(ascending=True)
        # Prepare the data to be represented
        data = [df[c].dropna().tolist() for c in avg_intensities.index]

        fig, ax = plt.subplots(figsize=(14, 13))
        bp = ax.boxplot(data, vert=False, patch_artist=True, notch=True,
                        whis=1.5)

        # Set styles for boxplot elements
        for box in bp['boxes']:
            box.set(color='black', linewidth=1.5)
            box.set(facecolor='lightgray')
        for whisker in bp['whiskers']:
            whisker.set(color='black', linewidth=1.5, linestyle='-')
        for cap in bp['caps']:
            cap.set(color='black', linewidth=1.5)
        for median in bp['medians']:
            median.set(color='red', linewidth=2)
        for flier in bp['fliers']:
            flier.set(marker='o', color='black', alpha=0.5)

        ax.set_yticklabels(avg_intensities.index)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        plt.tight_layout()
        return fig

    @staticmethod
    def get_comparative_data(
        df_model1: pd.DataFrame,
        df_model2 : pd.DataFrame,
        target: str
    ) -> pd.DataFrame:
        """Reads data from grouped CSV files for two models and
        prepares it for comparative bar chart representation.

        Args:
          df_model1: contains the aggregated experiment information
                     for the first model.
          df_model2: contains the aggregated experiment information
                     for the second model.
          target: value of the target class to represent.

        Returns:
          A DataFrame with the necessary information to create a
          comparative bar chart representing the results obtained for a
          method and target class of the two models.

          The DataFrame has three columns:
          - 'Region' indicates the name of the brain region.
          - 'Mean_model1' indicates the average intensity value for the
            first model.
          - 'Mean_model2' indicates the average intensity value for the
            second model.

        Raises:
          KeyError: if an invalid target has been specified.
        """
        # Check if the DataFrame contains the given target
        if target not in df_model1.columns or target not in df_model2.columns:
            raise KeyError('invalid target')

        # Extract mean data for each model
        mean_model1 = [r for r in df_model1.index if r.endswith(' mean')]
        mean_model2 = [r for r in df_model2.index if r.endswith(' mean')]

        # Select only mean rows for the specified target
        mean_data_model1 = df_model1.loc[mean_model1, str(target)]
        mean_data_model2 = df_model2.loc[mean_model2, str(target)]

        # Create a new DataFrame with the necessary data
        data = {
            'Region': [r.replace(' mean', '') for r in mean_model1],
            'Mean_model1': mean_data_model1.values,
            'Mean_model2': mean_data_model2.values
        }

        new_df = pd.DataFrame(data)
        new_df = new_df.sort_values(by='Mean_model1', ascending=True)
        return new_df

    @staticmethod
    def comparative_plot(
        df : pd.DataFrame,
        model1 : str='',
        model2 : str='',
        xlabel : str='',
        ylabel : str=''
    ) -> plt.Figure:
        """Creates a comparative bar chart of two models, showing the
        relevance of the ROIs for a method and class.

        Args:
          df: a DataFrame with the necessary data for the comparative
              plot.
              Read PlotGenerator.get_comparative_data() for more
              information.
          model1: indicates the label for the first model.
          model2: indicates the label for the second model.
          xlabel: indicates the label for the x-axis.
          ylabel: indicates the label for the y-axis.

        Returns:
          A figure representing a comparative bar chart between two
          models.

        Raises:
          ValueError: if the DataFrame is incorrect.
        """
        # Validate the DataFrame format
        columns = {'Region', 'Mean_model1', 'Mean_model2'}
        if not columns.issubset(df.columns):
            raise ValueError('incorrect dataframe')

        fig, ax = plt.subplots(figsize=(14, 13))
        y_positions = range(len(df))

        ax.barh([y - 0.2 for y in y_positions], df['Mean_model1'], height=0.4,
                color='red', label=model1, alpha=0.6)
        ax.barh([y + 0.2 for y in y_positions], df['Mean_model2'], height=0.4,
                color='blue', label=model2, alpha=0.6)

        # Set plot limits and labels
        ax.set_yticks(y_positions)
        ax.set_yticklabels(df['Region'])
        combined_min = min(df['Mean_model1'].min(), df['Mean_model2'].min())
        combined_max = max(df['Mean_model1'].max(), df['Mean_model2'].max())
        expand = combined_max - combined_min
        ax.set_xlim(combined_min - expand * 0.1, combined_max + expand * 0.2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()

        plt.tight_layout()
        return fig
