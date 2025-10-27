'''u_test_analysis'''


# IMPORT PACKAGES

import numpy as np
import pandas as pd

from prettytable import PrettyTable

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu

# Suppress Warnings
import warnings
warnings.filterwarnings('ignore')




# CREATE MANN-WHINEY U-TEST ANALYSIS CLASS

class UTestAnalysis():

    '''
    Class performs the Mann-Whitney U Test and returns the results.

    Returns:
    --------
    u_test_results: dictionary

    Outputs:
    --------
    box plot
    violin plot
    histogram
    ECDF plot
    tables
    '''

    # Define init Method
    def __init__(self, data=None, column=None, feature=None, unit_test=False):

        '''
        Method initializes class variables.
        
        Parameters:
        -----------
        self: The UTestAnalysis object
        self._data: Dataset to analyze
        self._column: Column for subgroups
        self._feature: The columns to measure
        unit_test: Determines if unit tests are being performed
        '''

        # Initialize Class Variables
        self._data = data
        self._column = column
        self._feature = feature
        self._unit_test = unit_test
    



    ############################## Getter Methods ##############################



    '''
    Methods get values from each class variable.
    '''

    # Getter for Data Variable
    @property
    def data(self):  
        return self._data


    # Getter for Column Variable
    @property
    def column(self):  
        return self._column
    

    # Getter for Feature Variable
    @property
    def feature(self):  
        return self._feature
    

    # Getter for Unit Test
    @property
    def unit_test(self):  
        return self._unit_test
    



    ############################## Setter Methods ##############################



    '''
    Methods set values for each class variable.
    '''

    # Define Setter for Loaded Data Variable
    @data.setter
    def data(self, dataframe):
        if isinstance(dataframe, pd.DataFrame):
            self._data = dataframe
        else:
            raise ValueError('Data must be a DataFrame.')
    



    # Define Setter for Column Variable
    @column.setter
    def column(self, column_name):
        if isinstance(column_name, str) and len(column_name) > 0:
            self._column = column_name
        else:
            raise ValueError('Column name must be a non-empty string.')
    
    


    # Define Setter for Feature Variable
    @feature.setter
    def feature(self, feature_column):
        if isinstance(feature_column, str) and len(feature_column) > 0:
            self._feature = feature_column
        else:
            raise ValueError('Feature column must be a non-empty string.')
    



    # Define Setter for Unit Test
    @unit_test.setter
    def unit_test(self, test_value):
        if isinstance(test_value, bool):
            self._unit_test = test_value
        else:
            raise ValueError('Test value must be either True or False.')




    ############################## Helper Methods ##############################



    # Define Method to Titlize a Column Name
    def titlize_column_name(self, column_name):

        '''
        Method creates a proper title from the column name.
        
        Parameters:
        -----------
        self: The UTestAnalysis object
        column_name: The name of the column to be titlized

        Returns:
        --------
        capitalized_name: string
        '''
        
        # Replace Underscore with Empty Space
        new_column_name = column_name.replace('_', ' ')

        # Capitalize First Letter in Every Word
        capitalized_name = new_column_name.title()
        
        # Return Capitalized Name
        return capitalized_name
    



    # Define Method to Create Table
    def create_table(self, dataframe, padding):

        '''
        Method creates a table from a dataframe for printing purposes.
        
        Parameters:
        -----------
        self: The UTestAnalysis object
        dataframe: The dataframe for the table

        Returns:
        --------
        table: table
        '''

        # Create a PrettyTable instance
        table = PrettyTable()

        # Set Text Alignment
        table.align = "l"

        # Set field names from DataFrame columns
        table.field_names = dataframe.columns.tolist()

        # Add rows from DataFrame to PrettyTable
        for index, row in dataframe.iterrows():
            table.add_row(row.tolist())

        table.float_format = "10.6"
        table.padding_width = padding

        # Return Table
        return table
    



    # Define Method to Split Data
    def split_data(self, data, column, feature):

        '''
        Method splits the dataframe into two series
        based on the column and feature.
        
        Parameters:
        -----------
        self: The UTestAnalysis object
        data: The dataset to analyze
        column: Column for subgroups
        feature: The column to measure

        Returns:
        --------
        dataset_1: pandas.Series
        dataset_2: pandas.Series
        '''

        # Select Key Column Values for Analysis
        group_list = data[column].unique()

        # Set Group Variable
        group_1 = group_list[0]
        group_2 = group_list[1]

        # Split the Data into Two Groups
        dataset_1 = data[data[column] == group_1][feature].dropna()
        dataset_2 = data[data[column] == group_2][feature].dropna()

        # Return Datasets
        return dataset_1, dataset_2
    



    # Define Method for ECDF Data
    def ecdf_data(self, data):
            
        '''
        Method calculates the data for the
        Empirical Cumulative Distribution Function.
        
        Parameters:
        -----------
        self: The UTestAnalysis object
        data: The dataset to analyze

        Returns:
        --------
        x: pandas.Series
        y: pandas.Series
        '''

        # Sort Data    
        x = np.sort(data)

        # Transform Data
        y = np.arange(1, len(data) + 1) / len(data)

        # Return Results
        return x, y




    ############################## Plot Methods ################################



    # Define Method for Box Plot
    def box_plot(self, data, column, feature, ax):

        '''
        Method creates a box plot comparing the two datasets.
        
        Parameters:
        -----------
        self: The UTestAnalysis object
        data: The dataset to analyze
        column: Column for subgroups
        feature: The column to measure
        ax: The subplot figure

        Returns:
        --------
        None

        Outputs:
        --------
        box plot
        '''

        # Create Title for Labels
        column_title = self.titlize_column_name(column)
        feature_title = self.titlize_column_name(feature)

        # Select Key Column Values for Palette
        group_list = data[column].value_counts().head(2).keys()
        group_1 = group_list[0]
        group_2 = group_list[1]

        # Create Color Palette
        palette = {group_1: "#3CB08B", group_2: "#2E3963"}

        # Create Boxplot
        sns.boxplot(x=column, hue=column, y=feature, data=data, palette=palette, ax=ax)

        # Add Border
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.25)
            ax.spines[axis].set_color("#010101")
        
        # Add Grid
        ax.grid(True, axis = 'y', color = '#010101', linestyle = '--', linewidth = 0.5)

        # Add Title and Labels
        ax.set_title(f'Box Plot of {feature_title} by {column_title}')
        ax.set_xlabel(column_title)
        ax.set_ylabel(feature_title)

        


    # Define Method for Violin Plot
    def violin_plot(self, data, column, feature, ax):

        '''
        Method creates a violin plot comparing the two datasets.
        
        Parameters:
        -----------
        self: The UTestAnalysis object
        data: The dataset to analyze
        column: Column for subgroups
        feature: The column to measure
        ax: The subplot figure

        Returns:
        --------
        None

        Outputs:
        --------
        violin plot
        '''

        # Create Title for Labels
        column_title = self.titlize_column_name(column)
        feature_title = self.titlize_column_name(feature)

        # Select Key Column Values for Palette
        group_list = data[column].value_counts().head(2).keys()
        group_1 = group_list[0]
        group_2 = group_list[1]

        # Create Color Palette
        palette = {group_1: "#3CB08B", group_2: "#2E3963"}

        # Create Violin Plot
        sns.violinplot(x=column, hue=column, y=feature, data=data, inner='quartile', palette=palette, ax=ax)

        # Add Border
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.25)
            ax.spines[axis].set_color("#010101")

        # Add Grid
        ax.grid(True, axis = 'y', color = '#010101', linestyle = '--', linewidth = 0.5)

        # Add Title and Labels
        ax.set_title(f'Violin Plot of {feature_title} by {column_title}')
        ax.set_xlabel(column_title)
        ax.set_ylabel(feature_title)

        


    # Define Method for Histogram Plot
    def histogram_plot(self, data, column, feature, ax):

        '''
        Method creates a histogram plot comparing the two datasets.
        
        Parameters:
        -----------
        self: The UTestAnalysis object
        data: The dataset to analyze
        column: Column for subgroups
        feature: The column to measure
        ax: The subplot figure

        Returns:
        --------
        None

        Outputs:
        --------
        histogram plot
        '''

        # Create Title for Labels
        column_title = self.titlize_column_name(column)
        feature_title = self.titlize_column_name(feature)

        # Select Key Column Values for Labels
        group_list = data[column].unique()

        # Split the Data into Two Groups
        dataset_1, dataset_2 = self.split_data(data, column, feature)

        # Create Histogram Plot for First Dataset
        sns.histplot(dataset_1, color='blue', alpha=0.5, label=group_list[0], kde=True, ax=ax)

        # Create Histogram Plot for Second Dataset
        sns.histplot(dataset_2, color='green', alpha=0.5, label=group_list[1], kde=True, ax=ax)

        # Add Border
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.25)
            ax.spines[axis].set_color("#010101")

        # Add Grid
        ax.grid(True, color = '#010101', linestyle = '--', linewidth = 0.5)

        # Add Title and Labels
        ax.set_title(f'Distribution of {feature_title} by {column_title}')
        ax.set_xlabel(feature_title)
        
        # Add Legend
        legend = ax.legend(shadow=True)
        legend.get_frame().set_edgecolor("#010101")




    # Define Method for Bar Plot
    def ecdf_plot(self, data, column, feature, p_value, ax):

        '''
        Method creates an Empirical Cumulative Distribution Function
        plot comparing the two datasets.
        
        Parameters:
        -----------
        self: The UTestAnalysis object
        data: The dataframe with the dataset stats
        column: Column for subgroups
        feature: The column to measure
        ax: The subplot figure

        Returns:
        --------
        None

        Outputs:
        --------
        step plot
        '''

        # Create Title for Labels
        feature_title = self.titlize_column_name(feature)

        # Select Key Column Values for Labels
        group_values = data[column].unique()

        # Split the Data into Two Groups
        dataset_1, dataset_2 = self.split_data(data, column, feature)

        # Calculate ECDF Data
        x1, y1 = self.ecdf_data(dataset_1)
        x2, y2 = self.ecdf_data(dataset_2)

        # Create ECDF Plot
        ax.step(x1, y1, color='blue', label=group_values[0], where='post')
        ax.step(x2, y2, color='green', label=group_values[1], where='post')

        # Determine P_value Significants
        if p_value < 0.05:
            p_sig = '\n(Significant)'
        else:
            p_sig = '\n(Not Significant)'

        # Create Label P-Value Annotation
        p_label = f'Mann-Whitney U Test\np = {p_value:.4f}' + p_sig

        # Add Text to Plot
        ax.text(0.95, 0.5, p_label, transform=ax.transAxes, ha='right', va='center', fontsize=12,
                    bbox={'facecolor':'white', 'edgecolor':'#010101', 'alpha':0.8, 'pad':5})
        
        # Add Border
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.25)
            ax.spines[axis].set_color("#010101")
        
        # Add Grid
        ax.grid(True, color = '#010101', linestyle = '--', linewidth = 0.5)
        
        # Add Title and Labels
        ax.set_title(f'Empirical Cumulative Distribution of {feature_title}')
        ax.set_xlabel(feature_title)
        ax.set_ylabel('Cumulative Probability')
        
        # Add Legend
        legend = ax.legend(shadow=True)
        legend.get_frame().set_edgecolor("#010101")



      
    # Define Method to Plot Data
    def plot_utest_analysis(self, data, column, feature, p_value):

        '''
        Method creates a figure displaying a series of plots
        for comparing the two datasets used in testing.
        
        Parameters:
        -----------
        self: The UTestAnalysis object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The column to measure
        p_value: The p-value from the Mann-Whitney U Test

        Returns:
        --------
        None

        Outputs:
        --------
        figure
        box plot
        violin plot
        step plot
        '''

        # Create Title for Labels
        column_title = self.titlize_column_name(column)
        feature_title = self.titlize_column_name(feature)

        # Set Figure Style
        sns.set_theme(style='whitegrid')
        plt.rcParams['axes.facecolor'] = "#F4F3F3FF"
        plt.rcParams['font.size'] = 12

        # Create Figure and Subplots
        _, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Create Plot 1: Box Plot
        ax1 = axes[0, 0]
        self.box_plot(data, column, feature, ax1)

        # Create Plot 2: Violin Plot
        ax2 = axes[0, 1]
        self.violin_plot(data, column, feature, ax2)

        # Create Plot 3: Histogram Plot
        ax3 = axes[1, 0]
        self.histogram_plot(data, column, feature, ax3)

        # Create Plot 4: ECDF Plot
        ax4 = axes[1, 1]
        self.ecdf_plot(data, column, feature, p_value, ax4)

        # Display Plots
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.suptitle(f'Mann-Whitney U Test: {feature_title} by {column_title}', fontsize=16, y=1.02)
        plt.show()


   

    ############################## Table Methods ###############################



    # Define Method for Plot Table
    def table_plot(self, dataframe, title, ax, col_widths=None, cell_location='right'):

        '''
        Method creates a table plot for the selected data.
        
        Parameters:
        -----------
        self: The UTestAnalysis object
        dataframe: The dataset to plot
        title: The title of the plot
        ax: The subplot figure
        col_widths: List of column widths
        cell_location: The test alignment

        Outputs:
        --------
        table
        '''

        # Create Data List
        data_list = []

        # Iterate Through Dataframe and Append to List
        for i in range(dataframe.shape[0]):
            data_list.append(dataframe.iloc[i].tolist())

        # Create Column Labels
        column_labels = dataframe.columns.tolist()

        # Create Colors List
        cell_colors = []
        column_colors = []
        row_colors = []

        # Iterate Through Data List and Append Cell Colors
        for data in data_list:
            colors = []
            for i in range(len(data)):
                colors.append('#ebeced')
            cell_colors.append(colors)
            row_colors.append('g')

        # Iterate Through Data List and Append Column Colors
        for i in range(len(data_list[0])):
            column_colors.append('#3f64a0')
        
        # Hide Axes
        ax.axis('off')
        
        # Create Table
        table = ax.table(cellText=data_list,
                        colWidths=col_widths,
                        cellLoc=cell_location,
                        cellColours=cell_colors,
                        colLabels=column_labels,
                        colColours=column_colors,
                        loc='upper center')
        
        # Set Number of Columns
        num_colums = len(column_labels)

        # Iterate through Column Header and Set Text Color
        for i in range(num_colums):
            cell = table[0, i]
            cell.get_text().set_color('#ebeced')

        # Set Table Properties
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        
        # Add Title
        ax.set_title(title, fontsize=12)
    



    # Define Method to Plot Table Data
    def plot_table_data(self, data, title, col_widths=None, cell_location='right', section_header=None):

        '''
        Method creates a figure and a table plot.
        
        Parameters:
        -----------
        self: The UTestAnalysis object
        data: the dataset to plot
        title: The title of the plot
        col_widths: List of column widths
        cell_location: The text alignment
        section_header: The suptitle text

        Outputs:
        --------
        figure
        table
        '''

        # Set Figure Style
        sns.set_style(style='whitegrid')

        # Create Subplots
        fig, ax = plt.subplots(1, 1, figsize=(12, 1))

        # Plot Table
        self.table_plot(data, title, ax, col_widths, cell_location)

        # Add Suptitle
        if section_header != None:
            fig.suptitle(f'{section_header}', fontsize=16, y=1.02)
        
        # Display Table
        plt.show()




    ############################## Analysis Methods ############################



    # Define Method to Calculate Statistical Data
    def stats_data(self, data, column, feature):

        '''
        Method creates a dataframe with the statistics
        of the two datasets.
        
        Parameters:
        -----------
        self: The UTestAnalysis object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The column to measure

        Returns:
        --------
        df_graph_data: dataframe

        Outputs:
        --------
        Group: The subgroup for the analysis
        Size: The size of the subgroups
        Mean: The means of the two datasets
        Median: The median of the two datasets
        Mode: The modes of the two datasets
        STD: The standard deviations of the two datasets
        IQR: The interquartile range of the two datasets
        '''

        # Select Key Column Values for Labels
        group_list = data[column].unique()

        # Split the Data into Two Groups
        dataset_1, dataset_2 = self.split_data(data, column, feature)

        # Calculate Descriptive Statistics
        n1, n2 = len(dataset_1), len(dataset_2)
        mean_1, mean_2 = dataset_1.mean(), dataset_2.mean()
        median_1, median_2 = dataset_1.median(), dataset_2.median()
        mode_1, mode_2 = dataset_1.mode()[0], dataset_2.mode()[0]

        # Calculate Standard Deviation
        std_1, std_2 = dataset_1.std(), dataset_2.std()

        # Calculate Interquartile Range
        iqr_1 = dataset_1.quantile(0.75) - dataset_1.quantile(0.25)
        iqr_2 = dataset_2.quantile(0.75) - dataset_2.quantile(0.25)

        # Create Dataframe for Stats Data
        df_stats_data = pd.DataFrame({
            'Group': [group_list[0], group_list[1]],
            'Size': [n1, n2],
            'Mean': [mean_1, mean_2],
            'Median': [median_1, median_2],
            'Mode': [mode_1, mode_2],
            'STD': [std_1, std_2],
            'IQR': [iqr_1, iqr_2]
        })

        # Return Dataframe
        return df_stats_data
    



    # Define Method for Effect Size
    def effect_size_analysis(self, data, column, feature, u_stat):

        '''
        Method calculates the z-score from the u-statistic
        to determine the effect size (r).
        
        Parameters:
        -----------
        self: The UTestAnalysis object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The column to measure
        U_stat: The statistical value from the Mann-Whitney U Test

        Returns:
        --------
        r_effect_size: float
        r_interpretation: string
        '''

        # Split the Data into Two Groups
        dataset_1, dataset_2 = self.split_data(data, column, feature)

        # Calculate Dataset Length
        n1, n2 = len(dataset_1), len(dataset_2)
        total_n =len(data)

        # Calculate Mean of U
        mean_u = (n1 * n2) / 2

        # Calculate Standard Deviation of U
        std_u = np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)

        # Calculate Z Score
        z_score = (u_stat - mean_u) / std_u

        # Calculate Effect Size (R)
        r_effect_size = abs(z_score) / np.sqrt(total_n)

        # Interpret Effect Size (R)
        if r_effect_size < 0.1:
            r_interpretation = "negligible"
        elif r_effect_size < 0.3:
            r_interpretation = "small"
        elif r_effect_size < 0.5:
            r_interpretation = "medium"
        else:
            r_interpretation = "large"

        # Return Results
        return r_effect_size, r_interpretation
    



    ############################## Print Methods ###############################



    # Define Methon for Printing Results
    def print_results(self, data, column, feature, u_stat, p_value):

        '''
        Method prints the statistical results of the Mann-Whitney U Test.
        
        Parameters:
        -----------
        self: The UTestAnalysis object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The column to measure
        u_stat: The u-statistic from the Mann-Whitney U Test
        p-value: The p-value from the Mann-Whitney U Test

        Returns:
        --------
        None

        Outputs:
        --------
        print statements
        stats table
        results table
        '''

        # Create Title for Labels
        feature_title = self.titlize_column_name(feature)

        # Calculate Descriptive Statistics
        df_stats = self.stats_data(data, column, feature)

        # Create Table From Stats Dataframe
        stats_table = self.create_table(df_stats, 1)

        # Format Columns to Six Decimal Places
        df_stats['Mean'] = df_stats['Mean'].apply(lambda x: f'{x:.6f}')
        df_stats['Median'] = df_stats['Median'].apply(lambda x: f'{x:.6f}')
        df_stats['Mode'] = df_stats['Mode'].apply(lambda x: f'{x:.6f}')
        df_stats['STD'] = df_stats['STD'].apply(lambda x: f'{x:.6f}')
        df_stats['IQR'] = df_stats['IQR'].apply(lambda x: f'{x:.6f}')

        # Create Lists from DataFrame Column Values
        group_1, group_2 = df_stats['Group'].tolist()

        # Calculate Effect Size (R)
        r_effect_size, r_interpretation = self.effect_size_analysis(data, column, feature, u_stat)

        # Create Dataframe for Results Data
        df_results_data = pd.DataFrame({
            'Statistic': [
                'U Statistic',
                'P_Value',
                'Effect Size (R)',
                'R Interpretation'
            ],
            'Value': [
                f'{u_stat:.2f}',
                f'{p_value:.6f}',
                f'{r_effect_size:.4f}',
                f'{r_interpretation} effect',
            ]
        })

        # Create Table From Stats Dataframe
        results_table = self.create_table(df_results_data, 1)

        # Determine Final U-Test Assessment
        if p_value < 0.05:
            u_interpretation = f'There is a statistically significant difference in {feature_title} between {group_1} and {group_2}'
        else:
            u_interpretation = f'There is no statistically significant difference in {feature_title} between {group_1} and {group_2}'
        
        # Create Dataframe for U-Test Interpretation
        df_interpretation = pd.DataFrame({
            'Interpretation': [f'{u_interpretation}']
        })


        # Print Stats Table
        stats_title = f'Comparing {feature_title} between {group_1} and {group_2}'
        stats_col_widths = [0.15, 0.1, 0.15, 0.15, 0.15, 0.15, 0.15]
        self.plot_table_data(df_stats.round(6), stats_title, stats_col_widths, 'left')

        
        # Print Results Table
        results_title = f'Mann-Whitney U Test Results'
        results_col_widths = [0.5, 0.5]
        self.plot_table_data(df_results_data, results_title, results_col_widths, 'left')


        # Print Final T-Test Interpretation
        final_title = 'Overall Assessment of Mann-Whitney U Test Analysis'
        final_col_widths = [1.0]
        self.plot_table_data(df_interpretation, final_title, final_col_widths, 'center')

        


    ############################## Main Method #################################



    # Define Method for Mann-Whitney U Test
    def mannwhitney_utest_analysis(self, column=None, feature=None, unit_test=None):

        '''
        Method performs the Mann-Whitney U Test and returns the results.
        
        Parameters:
        -----------
        self: The UTestAnalysis object
        column: Column for subgroups
        feature: The column to measure
        unit_test: Determines if unit tests are being performed

        Returns:
        --------
        u_test_results: dictionary

        Outputs:
        --------
        box plot
        violin plot
        histogram
        ECDF plot
        tables
        '''

        # Set Data Variable
        data = self._data

        # Set Column Variable
        if column == None:
            column = self._column

        # Set Feature Variable
        if feature == None:
            feature = self._feature
        
        # Set Unit Test Variable
        if unit_test == None:
            unit_test = self._unit_test

        # Split the Data into Two Groups
        dataset_1, dataset_2 = self.split_data(data, column, feature)
        
        # Calculate Descriptive Statistics for Each Group
        df_stats = self.stats_data(data, column, feature)

        # Create Lists from DataFrame Column Values
        group_1, group_2 = df_stats['Group'].tolist()
        n1, n2 = df_stats['Size'].tolist()

        # Perform Mann-Whitney U Test
        u_stat, p_value = mannwhitneyu(dataset_1, dataset_2, alternative='two-sided')

        # Calculate Effect Size (R)
        r_effect_size, r_interpretation = self.effect_size_analysis(data, column, feature, u_stat)
        
        # Plot Data
        if unit_test == False:
            self.plot_utest_analysis(data, column, feature, p_value)

        # Print Results
        if unit_test == False:
            self.print_results(data, column, feature, u_stat, p_value)

        # Create Results Dictionary
        u_test_results = {
            'group_1': group_1,
            'group_2': group_2,
            'n1': n1,
            'n2': n2,
            'u_statistic': u_stat,
            'p_value': p_value,
            'r_effect_size': r_effect_size,
            'r_interpretation': r_interpretation
        }

        # Return Results Dictionary
        return u_test_results















# End of Page
