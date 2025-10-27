'''parametric_nonparametric_analysis'''


# IMPORT PACKAGES

import numpy as np
import pandas as pd

import math

from prettytable import PrettyTable

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import mannwhitneyu

from distribution_analysis import DistributionAnalysis
from transformation_analysis import TransformationAnalysis
from t_test_analysis import TTestAnalysis
from u_test_analysis import UTestAnalysis




# CREATE PARAMETRIC NONPARAMETRIC ANALYSIS CLASS

class ParametricNonparametricAnalysis():

    '''
    Class performs analysis of parametric and non-parametric test results.

    Returns:
    --------
    parametric_nonparametric_results: list

    Outputs:
    --------
    figure
    tables
    '''

    # Define init Method
    def __init__(self, data=None, column=None, feature=None, 
                 target=None, unit_test=False):

        '''
        Method initializes class variables.
        
        Parameters:
        -----------
        self: The DistributionAnalysis object
        self._data: Dataset to analyze
        self._column: Column for subgroups
        self._feature: The columns to measure
        self._target: Column of the target of comparison
        unit_test: Determines if unit tests are being performed
        '''

        # Initialize Class Variables
        self._data = data
        self._column = column
        self._feature = feature
        self._target = target
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
    

    # Getter for Target Variable
    @property
    def target(self):  
        return self._target
    

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
    



    # Define Setter for Target Variable
    @target.setter
    def target(self, target_column):
        if isinstance(target_column, str) and len(target_column) > 0:
            self._target = target_column
        else:
            raise ValueError('Target column must be a non-empty string.')
    



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
        self: The ParametricNonparametricAnalysis object
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
        self: The ParametricNonparametricAnalysis object
        dataframe: The dataframe for the table
        padding: The padding around the text

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
        self: The ParametricNonparametricAnalysis object
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
    



    ############################## Table Methods ###############################



    # Define Method to Plot Table Title
    def plot_table_title(self, title):

        '''
        Method creates an empty figure with a title.
        
        Parameters:
        -----------
        self: The ParametricNonparametricAnalysis object
        title: The title of the plot

        Outputs:
        --------
        figure
        table
        '''

        # Create Subplots
        fig, ax = plt.subplots(1, 1, figsize=(11, .1))

        # Hide Axes
        ax.axis('off')

        # Add Suptitle
        fig.suptitle(f'{title}', fontsize=16, y=1.02, x=0.6, ha='center')
                
        # Display Table
        plt.show()




    # Define Method for Plot Table
    def table_plot(self, dataframe, title, ax, col_widths=None, cell_location='right'):

        '''
        Method creates a table plot for the selected data.
        
        Parameters:
        -----------
        self: The ParametricNonparametricAnalysis object
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

        # Iterate through Column Heade and Set Text Color
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
        self: The ParametricNonparametricAnalysis object
        data: The dataset to plot
        title: The title of the plot
        col_widths: List of column widths
        cell_location: The test alignment
        section_header: The suptitle text

        Outputs:
        --------
        figure
        table
        '''

        # Set Figure Style
        sns.set_style(style='whitegrid')

        # Set Total Plots
        total_plots = len(data)

        # Set Number of Columns
        if total_plots > 1:
            columns = 2
        else:
            columns = 1
        
        # Set Number of Rows
        rows = total_plots / columns
        rows = math.ceil(rows)

        # Create Subplots
        if total_plots == 1:
            fig, ax1 = plt.subplots(rows, columns, figsize=(12, 1))
        else:
            fig, axs = plt.subplots(rows, columns, figsize=(13, 1))

        # Set AX1 to List
        if total_plots == 1:
            axs = [ax1]

        # Plot Tables
        for i in range(len(data)):
            self.table_plot(data[i], title[i], axs[i], col_widths, cell_location)

        # Add Suptitle
        if section_header != None:
            fig.suptitle(f'{section_header}', fontsize=16, y=1.02)
        
        # Create Padding
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, 
                    top=0.9, wspace=0.4,hspace=0.4)
        
        # Display Table
        plt.show()




    ############################## Analysis Methods ############################



    # Define Method for Transformed Data
    def transformed_data(self, data, column, feature):

        '''
        Method performs transformations on the data to normalize.
        
        Parameters:
        -----------
        self: The ParametricNonparametricAnalysis object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The columns to measure

        Returns:
        --------
        xform: string
        df_xform_data: dataframe
        '''

        # Select Key Column Values for Analysis
        group_list = data[column].value_counts().head(8).keys()

         # Set Group Variables
        group_1 = group_list[0]
        group_2 = group_list[1]

        # Filter Data
        filtered_data_1 = data[data.apply(lambda row: row[column] == group_1, axis=1)]
        filtered_data_2 = data[data.apply(lambda row: row[column] == group_2, axis=1)]

        # Create Analysis Objects
        ta_1 = TransformationAnalysis(filtered_data_1)
        ta_2 = TransformationAnalysis(filtered_data_2)

        # Perform Data Transformations
        df_summary_1, data_list_1 = ta_1.data_transformations(filtered_data_1, feature)
        df_summary_2, data_list_2 = ta_2.data_transformations(filtered_data_2, feature)

        # Determine Best Transformations
        xform, data_xform_1, _, _ = ta_1.best_transformation(df_summary_1, data_list_1)
        xform, data_xform_2, _, _ = ta_2.best_transformation(df_summary_2, data_list_2)

        # Create Dataframes
        df_xform_1 = pd.DataFrame(data_xform_1, columns=[f'{feature}'])
        df_xform_2 = pd.DataFrame(data_xform_2, columns=[f'{feature}'])

        # Add Dataset Names
        df_xform_1[f'{column}'] = group_1
        df_xform_2[f'{column}'] = group_2

        # Combine Dataframes
        df_xform_data = pd.concat([df_xform_1, df_xform_2], axis=0)

        # Return results
        return xform, df_xform_data
    



    # Define Method for Normality Analysis
    def normality_analysis(self, data, column, feature, target):

        '''
        Method performs an analysis of data to determine normality.
        
        Parameters:
        -----------
        self: The ParametricNonparametricAnalysis object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The columns to measure
        target: The column to subdivide data into groups

        Returns:
        --------
        normality_results: dictionary
        '''

        # Create Analysis Objects
        da = DistributionAnalysis(data)
        ut = UTestAnalysis(data)

        # Calculate Descriptive Statistics for Each Group
        df_stats = ut.stats_data(data, column, feature)

        # Create Lists from DataFrame Column Values
        group_1, group_2 = df_stats['Group'].tolist()

        # Filter Data
        filtered_data_1 = data[data.apply(lambda row: row[column] == group_1, axis=1)]
        filtered_data_2 = data[data.apply(lambda row: row[column] == group_2, axis=1)]

        # Perform Shapiro-Wilk Test
        _, shapiro_1 = da.shapiro_wilk_test(filtered_data_1, feature, target)
        _, shapiro_2 = da.shapiro_wilk_test(filtered_data_2, feature, target)

        # Create Results Dictionary
        normality_results = {
            'group_1': group_1,
            'group_2': group_2,
            'shapiro_1': shapiro_1,
            'shapiro_2': shapiro_2,
            'df_stats': df_stats
        }

        # Return Results
        return normality_results



    # Define Method for Parametric Analysis
    def parametric_analysis(self, data, column, feature):

        '''
        Method performs a parametric analysis of data to determine statistical significance.
        
        Parameters:
        -----------
        self: The ParametricNonparametricAnalysis object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The columns to measure

        Returns:
        --------
        parametric_results: dictionary
        '''

        # Create Analysis Object
        tt = TTestAnalysis(data)

        # Split the Data into Two Groups
        dataset_1, dataset_2 = self.split_data(data, column, feature)

        # Perform Equal Variance Analysis
        levene_p, equal_variance = tt.equal_variance_analysis(data, column, feature)       

        # Perform the T-Test
        t_stat, tt_p_value = stats.ttest_ind(dataset_1, dataset_2, equal_var=equal_variance)

        # Calculate Effect Size
        d, d_interpretation = tt.effect_size(data, column, feature)

        # Create Results Dictionary
        parametric_results = {
            'statistic': t_stat,
            'p_value': tt_p_value,
            'size_effect': d,
            'interpretation': d_interpretation,
            'levene_p': levene_p,
            'equal_variance': equal_variance
        }

        # Return Results
        return parametric_results
    



    # Define Method for Non-Parametric Analysis
    def non_parametric_analysis(self, data, column, feature):

        '''
        Method performs a non-parametric analysis of data to determine statistical significance.
        
        Parameters:
        -----------
        self: The ParametricNonparametricAnalysis object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The columns to measure

        Returns:
        --------
        non_parametric_results: dictionary
        '''

        # Create Analysis Objects
        ut = UTestAnalysis(data)

        # Split the Data into Two Groups
        dataset_1, dataset_2 = self.split_data(data, column, feature)

        # Perform Mann-Whitney U Test
        u_stat, ut_p_value = mannwhitneyu(dataset_1, dataset_2, alternative='two-sided')

        # Calculate Effect Size (R)
        r_effect_size, r_interpretation = ut.effect_size_analysis(data, column, feature, u_stat)

        # Create Results Dictionary
        non_parametric_results = {
            'statistic': u_stat,
            'p_value': ut_p_value,
            'size_effect': r_effect_size,
            'interpretation': r_interpretation
        }

        # Return Results
        return non_parametric_results



    ############################## Print Methods ###############################



    # Define Method for Printing Normality Analysis Results
    def print_normality_analysis(self, group, shapiro_results, print_results=True):

        '''
        Method prints the statistical results of the Shapiro-Wilk normality test
        and returns the results in a dataframe.
        
        Parameters:
        -----------
        self: The ParametricNonparametricAnalysis object
        group: The subgroup of the analysis
        shapiro_results: Dictionary with the results of the normality tests
        print_results: Prints results if True

        Outputs:
        --------
        print statements
        results table

        Returns:
        --------
        df_normality_results: dataframe
        normality_title: string
        '''

        # Set Results Variables
        statistic, p_value, _ = shapiro_results

        # P_Value Interpretation
        p_interpretation = 'Normal Distribution' if p_value > 0.05 else 'Non-Normal Distribution'

        # Create Dataframe for Normality Results
        df_normality_results = pd.DataFrame({
            'Statistic': [
                'Test',
                'Statistic',
                'P_Value',
                'Interpretation'
            ],
            'Value': [
                f'Shapiro-Wilk',
                f'{statistic:.6f}',
                f'{p_value:.6f}',
                f'{p_interpretation}'
            ]
        })

        # Create Table From Results Dataframe
        results_table = self.create_table(df_normality_results, 2)

        # Create Title Variable
        normality_title = f'Analysis of Normality Test on {group}'

        # Print Results
        if print_results == True:
            print(normality_title)
            print(results_table)

        # Return Normality Results
        return df_normality_results, normality_title




    # Define Method for Printing Equal Variance Results
    def print_equal_variance(self, parametric_results, data_form, print_results=True):

        '''
        Method prints the statistical results of the Equal Variances Test
        embedded in the Parametric Test and returns the results in a dataframe.
        
        Parameters:
        -----------
        self: The ParametricNonparametricAnalysis object
        parametric_results: Dictionary with the results of the Parametric Test
        data_form: The form of the data used in testing
        print_results: Prints results if True

        Embedded Parameters:
        --------------------
        p-value: The p-value from the Equal Variances Test
        equal_variance: The boolean value from the Equal Variances Test

        Outputs:
        --------
        print statements
        results table

        Returns:
        --------
        df_ev_results: dataframe
        ev_title: string
        '''

        # Set Equal Variance Variable
        p_value = parametric_results['levene_p']
        equal_variance = parametric_results['equal_variance']

        # Equal Variance Interpretation
        ev_interpretation = 'Equal Variances' if equal_variance else 'Unequal Variances'

        # Create Dataframe for Equal Variance Results
        df_ev_results = pd.DataFrame({
            'Statistic': [
                'Test',
                'P_Value',
                'Interpretation',
                '  ',
                ' '
            ],
            'Value': [
                f"Levene's Test",
                f'{p_value:.6f}',
                f'{ev_interpretation}',
                '  ',
                ' '
            ]
        })

        # Create Table From Results Dataframe
        results_table = self.create_table(df_ev_results, 3)

        # Create Title Variable
        ev_title = f'Analysis of Equal Variance Test - {data_form}'

        # Print Results
        if print_results == True:
            print(ev_title)
            print(results_table)
        
        # Return Equal Variance Results
        return df_ev_results, ev_title




    # Define Method for Printing Parametric Test Results
    def print_parametric_ressults(self, parametric_results, data_form, print_results=True):

        '''
        Method prints the statistical results of the Parametric Test
        and returns the results in a dataframe.
        
        Parameters:
        -----------
        self: The ParametricNonparametricAnalysis object
        parametric_results: Dictionary with the results of the Parametric Test
        data_form: The form of the data used in testing
        print_results: Prints results if True

        Embedded Parameters:
        --------------------
        t_stat: The t-statistic from the Parametric Test
        p-value: The p-value from the Parametric Test
        d: The effect size value

        Outputs:
        --------
        print statements
        results table

        Returns:
        --------
        df_parametric_results: dataframe
        parametric_title: string
        '''

        # Set Parametric Test Variables
        t_stat = parametric_results['statistic']
        p_value = parametric_results['p_value']
        d = parametric_results['size_effect']

        # P_Value Interpretation
        p_interpretation = 'Significant' if p_value < 0.05 else 'Not significant'

        # Create Dataframe for Parametric Results
        df_parametric_results = pd.DataFrame({
            'Statistic': [
                'Test',
                'T Statistic',
                'P_Value',
                'Effect Size D',
                'Interpretation'
            ],
            'Value': [
                f'Independent T-Test',
                f'{t_stat:.4f}',
                f'{p_value:.6f}',
                f'{d:.4f}',
                f'{p_interpretation}'
            ]
        })

        # Create Table From Results Dataframe
        results_table = self.create_table(df_parametric_results, 3)

        # Create Title Variable
        parametric_title = f'Analysis of Parametric Test - {data_form}'

        # Print Results
        if print_results == True:
            print(parametric_title)
            print(results_table)

        # Return Parametric Results
        return df_parametric_results, parametric_title




    # Define Method for Printing Non-Parametric Test Results
    def print_nonparametric_ressults(self, non_parametric_results, print_results=True):

        '''
        Method prints the statistical results of the Non-Parametric Test
        and returns the results in a dataframe.
        
        Parameters:
        -----------
        self: The ParametricNonparametricAnalysis object
        non_parametric_results: Dictionary with the results of the Non-Parametric test
        print_results: Prints results if True

        Embedded Parameters:
        --------------------
        t_stat: The t-statistic from the Non-Parametric Test
        p-value: The p-value from the Non-Parametric Test
        r: The effect size value

        Outputs:
        --------
        print statements
        results table

        Returns:
        --------
        df_nonparametric_results: dataframe
        nonparametric_title: string
        '''

        # Set Non-Parametric Test Variables
        u_stat = non_parametric_results['statistic']
        p_value = non_parametric_results['p_value']
        r = non_parametric_results['size_effect']

        # P_Value Interpretation
        p_interpretation = 'Significant' if p_value < 0.05 else 'Not significant'

        # Create Dataframe for Non-Parametric Results
        df_nonparametric_results = pd.DataFrame({
            'Statistic': [
                'Test',
                'U Statistic',
                'P_Value',
                'Effect Size R',
                'Interpretation',
                ' '
            ],
            'Value': [
                f'Mann-Whitney U Test',
                f'{u_stat:.4f}',
                f'{p_value:.6f}',
                f'{r:.4f}',
                f'{p_interpretation}',
                ' '
            ]
        })

        # Create Table From Results Dataframe
        results_table = self.create_table(df_nonparametric_results, 3)

        # Create Title Variable
        nonparametric_title = f'Analysis of Non-Parametric Test'

        # Print Results
        if print_results == True:
            print(nonparametric_title)
            print(results_table)

        # Return Non-Parametric Results
        return df_nonparametric_results, nonparametric_title
    



    # Define Method to Print Statistics
    def print_statistics(self, normality_results, print_results=True):

        '''
        Method prints the results of the statistical analysis in the Normality Test
        and returns the results in a dataframe.
        
        Parameters:
        -----------
        self: The ParametricNonparametricAnalysis object
        normality_results: Dictionary with the results of the Normality Test
        print_results: Prints results if True

        Embedded Parameters:
        --------------------
        df_stats: The dataframe with the statistics
        group_1 = The name of the first data group
        group_2 = The name of the second data group

        Outputs:
        --------
        print statements
        results table

        Returns:
        --------
        df_stats: dataframe
        stats_title: string
        '''

        # Set Variables
        df_stats = normality_results['df_stats']
        group_1 = normality_results['group_1']
        group_2 = normality_results['group_2']

        # Transpose Stats Dataframe
        df_stats = df_stats.transpose()

        # Reset Index
        df_stats = df_stats.reset_index()

        # Assign First Row as Column Names
        df_stats.columns = df_stats.iloc[0]

        # Remove First Row from Dataframe
        df_stats = df_stats[1:]

        # Get First Column Name
        first_col_name = df_stats.columns[0]

        # Rename First Column
        df_stats = df_stats.rename(columns={first_col_name: 'Statistic'})


        # Create Function to Format Decimals
        def format_decimals(x):
            if isinstance(x, (int)):
                return f'{x:.0f}'
            elif isinstance(x, (float)):
                return f'{x:.6f}'
            else:
                return x


        # Format Columns to Six Decimal Places
        df_stats[f'{group_1}'] = df_stats[f'{group_1}'].apply(format_decimals)
        df_stats[f'{group_2}'] = df_stats[f'{group_2}'].apply(format_decimals)

        # Create Table From Stats Dataframe
        stats_table = self.create_table(df_stats, 3)

        # Create Title Variable
        stats_title = f'Statistical Analysis for {group_1} and {group_2}'

        # Print Statistics
        if print_results == True:
            print(stats_title)
            print(stats_table)

        # Return Statistics Results
        return df_stats, stats_title




    # Define Method for Printing Final Assessment Results
    def print_final_assessment(self, p_value_orig, p_value_xform, p_value_nonparam, print_results=True):

        '''
        Method prints the final assessment of the Parametric Test
        Non-Parametric Test comparison, returning the results in a dataframe.
        
        Parameters:
        -----------
        self: The ParametricNonparametricAnalysis object
        p_value_orig: The p-value from the Parametric Test with original data
        p_value_xform: The p-value from the Parametric Test with transformed data
        p_value_nonparam: The p-value from the Non-Parametric Test
        print_results: Prints results if True

        Outputs:
        --------
        print statements
        results table

        Returns:
        --------
        df_interpretation: dataframe
        assessment_title: string
        '''

        # Set Variables
        sig_results = 0
        sig_orig = False
        sig_xform = False
        sig_nonparam = False

        if p_value_orig < 0.05:
            sig_results += 1
            sig_orig = True

        if p_value_xform < 0.05:
            sig_results += 1
            sig_xform = True
        
        if p_value_nonparam < 0.05:
            sig_results += 1
            sig_nonparam = True

        # Determine Assessment Interpretation
        if sig_results == 3:
            a_interpretation = 'All tests produce similar results with respect to a statistical significance'
        elif sig_results == 2:
            a_interpretation = 'Two of the three tests produce similar results with respect to a statistical significance'
        elif sig_results == 1:
            a_interpretation = 'Two of the three tests produce similar results with respect to no statistical significance'
        elif sig_results == 0:
            a_interpretation = 'All tests produce similar results with respect to no statistical significance'
        else:
            a_interpretation = 'All tests produce different results with respect to statistical significance'


        # Determine P-Value Interpretation if All Equal
        if sig_orig == sig_xform and sig_xform == sig_nonparam:
            if (abs(p_value_orig - p_value_xform) < 0.01) and  (abs(p_value_xform - p_value_nonparam) < 0.01):
                p_interpretation = 'The p-values are comparible with each other in all tests'
            elif p_value_orig < p_value_xform and p_value_orig < p_value_nonparam:
                p_interpretation = 'The parametric with original data produced a more significant result'
            elif p_value_xform < p_value_orig and p_value_xform < p_value_nonparam:
                p_interpretation = 'The parametric with transformed data produced a more significant result'
            elif p_value_nonparam < p_value_orig and p_value_nonparam < p_value_xform:
                p_interpretation = 'The non-parametric produced a more significant result'
            elif p_value_orig < p_value_nonparam and p_value_xform < p_value_nonparam:
                p_interpretation = 'Both parametric produced a more significant result'
            else:
                p_interpretation = 'The test produced a mixed of significant results'


        # Determine P-Value Interpretation if Orig and Xform are Equal
        elif sig_orig == sig_xform:
            if abs(p_value_orig - p_value_xform) < 0.01:
                p_interpretation = 'The p-values are comparible with each other in both parametric tests'
            elif p_value_orig < p_value_xform:
                p_interpretation = 'The parametric with original data produced a more significant result'
            else:
                p_interpretation = 'The parametric with transformed data produced a more significant result'
        

        # Determine P-Value Interpretation if Orig and Non-Param are Equal
        elif sig_orig == sig_nonparam:
            if abs(p_value_orig - p_value_nonparam) < 0.01:
                p_interpretation = 'The p-values are comparible with each other in both parametric original and non-parametric tests'
            elif p_value_orig < p_value_nonparam:
                p_interpretation = 'The parametric with original data produced a more significant result'
            else:
                p_interpretation = 'The non-parametric produced a more significant result'
        

        # Determine P-Value Interpretation if Xform and Non-Param are Equal
        elif sig_xform == sig_nonparam:
            if abs(p_value_xform - p_value_nonparam) < 0.01:
                p_interpretation = 'The p-values are comparible with each other in both parametric transformed and non-parametric tests'
            elif p_value_xform < p_value_nonparam:
                p_interpretation = 'The parametric with transformed data produced a more significant result'
            else:
                p_interpretation = 'The non-parametric produced a more significant result'
        
        # Determine P-Value Interpretation if None are Equal
        else:
            p_interpretation = 'The p-values are not comparible with each other in any of the tests'


        # Create Dataframe for Interpretations
        df_interpretation = pd.DataFrame({
            'Interpretation': [f'{a_interpretation}', f'{p_interpretation}']
        })

        # Create Table From Interpretation Dataframe
        assessment_table = self.create_table(df_interpretation, 3)

        # Create Title Variable
        assessment_title = f'Overall Assessment of Parametric vs Non-Parametric Analysis'

        # Print Assessment
        if print_results == True:
            print(assessment_title)
            print(assessment_table)
        
        # Return Assessment Results
        return df_interpretation, assessment_title




    ############################## Print Run Methods ###########################



    # Define Method to Plot Normality Analysis
    def plot_normality_analysis(self, normality_results):

        '''
        Method plots tables with the results of the Normality Tests.
        
        Parameters:
        -----------
        self: The ParametricNonparametricAnalysis object
        normality_results: Dictionary with the results of the Normality Test

        Embedded Parameters:
        --------------------
        group_1 = The name of the first data group
        group_2 = The name of the second data group
        shapiro_1 = The first result Shapiro-Wilk normality test
        shapiro_2 = The second result Shapiro-Wilk normality test

        Outputs:
        --------
        normality tables
        '''

        # Set Normality Variables
        group_1 = normality_results['group_1']
        group_2 = normality_results['group_2']
        shapiro_1 = normality_results['shapiro_1']
        shapiro_2 = normality_results['shapiro_2']

        # Perform Print Normality Analysis
        normality_data_1, normality_title_1 = self.print_normality_analysis(group_1, shapiro_1, False)
        normality_data_2, normality_title_2 = self.print_normality_analysis(group_2, shapiro_2, False)

        # Create Normality Lists
        normality_datasets = [normality_data_1, normality_data_2]
        normality_titles = [normality_title_1, normality_title_2]

        # Plot Normality Tables
        normality_col_widths = [0.5, 0.5]
        self.plot_table_data(normality_datasets, normality_titles, normality_col_widths, 'left')
    



    # Define Method to Plot Parametric Analysis
    def plot_parametric_analysis(self, parametric_results, data_form):

        '''
        Method plots tables with the results of the Equal Variances and Parametric Tests.
        
        Parameters:
        -----------
        self: The ParametricNonparametricAnalysis object
        parametric_results: Dictionary with the results of the Parametric Test
        normality_results: Dictionary with the results of the Normality Test
        data_form: The form of the data used in testing

        Outputs:
        --------
        equal variances table
        parametric table
        '''

        # Perform Print Equal Variance Results
        ev_data, ev_title = self.print_equal_variance(parametric_results, data_form, print_results=False)

        # Perform Print Parametric Test Results
        parametric_data, parametric_title = self.print_parametric_ressults(parametric_results, data_form, print_results=False)

        # Create Parametric Lists
        parametric_datasets = [ev_data, parametric_data]
        parametric_titles = [ev_title, parametric_title]

        # Plot Parametric Tables
        parametric_col_widths = [0.5, 0.5]
        self.plot_table_data(parametric_datasets, parametric_titles, parametric_col_widths, 'left')
    



    # Define Method to Plot Non-Parametric Analysis
    def plot_nonparametric_analysis(self, non_parametric_results, normality_results):

        '''
        Method plots tables with the results of the Non-Parametric Test
        and the statistical analysis embedded in the Normality Test.
        
        Parameters:
        -----------
        self: The ParametricNonparametricAnalysis object
        non_parametric_results: Dictionary with the results of the Non-Parametric Test
        normality_results: Dictionary with the results of the Normality Test

        Outputs:
        --------
        non-parametric table
        stats table
        '''

        # Perform Print Non-Parametric Test Results
        non_parametric_data, non_parametric_title = self.print_nonparametric_ressults(non_parametric_results, print_results=False)

        # Perform Print Statistical Analysis
        stats_data, stats_title = self.print_statistics(normality_results, print_results=False)

        # Create Parametric Lists
        non_parametric_datasets = [non_parametric_data, stats_data]
        non_parametric_titles = [non_parametric_title, stats_title]

        # Plot Non-Parametric Tables
        non_parametric_col_widths = None
        self.plot_table_data(non_parametric_datasets, non_parametric_titles, non_parametric_col_widths, 'left')
    



    # Define Method to Plot Final Analysis
    def plot_final_anlysis(self, original_results, xform_results, non_parametric_results):

        '''
        Method plots tables with the final interpretation analysis of the Parametric
        and Non-Parametric Tests.
        
        Parameters:
        -----------
        self: The ParametricNonparametricAnalysis object
        original_results: Dictionary with the results of the Parametric Test with original data
        xform_results: Dictionary with the results of the Parametric Test with transformed data
        non_parametric_results: Dictionary with the results of the Non-Parametric Test

        Embedded Parameters:
        --------------------
        p_value_orig: The p-value from the Parametric Test with original data
        p_value_xform: The p-value from the Parametric Test with transformed data
        p_value_nonparam: The p-value from the Non-Parametric Test

        Outputs:
        --------
        assessment table
        '''

        # Set P-Value Variables
        p_value_orig = original_results['p_value']
        p_value_xform = xform_results['p_value']
        p_value_nonparam = non_parametric_results['p_value']

        # Perform Print Final Assesment Results
        assessment_data, assessment_title = self.print_final_assessment(p_value_orig, p_value_xform, p_value_nonparam, print_results=False)

        # Plot Final Assesment Table
        assessment_col_widths = None
        self.plot_table_data([assessment_data], [assessment_title], assessment_col_widths, 'left')




    # Define Method to Plot Analysis Results
    def plot_analysis_results(self, normality_results, original_results, 
                              xform_results, non_parametric_results, feature):

        '''
        Method plots all tables with the results of the Parametric
        and Non-Parametric analysis for the given feature.
        
        Parameters:
        -----------
        self: The ParametricNonparametricAnalysis object
        normality_results: Dictionary with the results of the Normality Test
        original_results: Dictionary with the results of the Parametric Test with original data
        xform_results: Dictionary with the results of the Parametric Test with transformed data
        non_parametric_results: Dictionary with the results of the Non-Parametric Test
        feature: The columns to measure

        Outputs:
        --------
        section header
        normality tables
        equal variances tables
        parametric tables
        non-parametric table
        stats table
        assessment table
        '''

        # Create Title for Labels
        feature_title = self.titlize_column_name(feature)

        # Create Section Header
        section_header = f'\nComparing Parametric and Non-Parametric Tests for {feature_title}'

        # Plot Section Header
        self.plot_table_title(section_header)

        # Plot Normality Analysis
        self.plot_normality_analysis(normality_results)

        # Plot Parametric Analysis - Original Data
        self.plot_parametric_analysis(original_results, 'Original')

        # Plot Parametric Analysis - Transformed Data
        self.plot_parametric_analysis(xform_results, 'Transformed')

        # Plot Non-Parametric Analysis
        self.plot_nonparametric_analysis(non_parametric_results, normality_results)

        # Plot Final Analysis
        self.plot_final_anlysis(original_results, xform_results, non_parametric_results)




    ############################## Main Method #################################



    # Define Method for Parametric Non-Parametric Analysis
    def parametric_nonparametric_analysis(self, column=None, feature=None, target=None, unit_test=None):

        '''
        Method performs an analysis of parametric and non-parametric test results.
        
        Parameters:
        -----------
        self: The ParametricNonparametricAnalysis object
        column: Column for subgroups
        feature: The columns to measure
        target: The column to subdivide data into groups
        unit_test: Determines if unit tests are being performed

        Returns:
        --------
        parametric_nonparametric_results: dictionary

        Outputs:
        --------
        figure
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
        
        # Set Target Variable
        if target == None:
            target = self._target
        
        # Set Unit Test Variable
        if unit_test == None:
            unit_test = self._unit_test

        # Transform Data
        _, xform_data = self.transformed_data(data, column, feature)

        # Perform Normality Analysis on Data
        normality_results = self.normality_analysis(data, column, feature, target)

        # Perform Parametric Analysis on Data
        original_results = self.parametric_analysis(data, column, feature)
        xform_results = self.parametric_analysis(xform_data, column, feature)

        # Perform Non-Parametric Analysis on Data
        non_parametric_results = self.non_parametric_analysis(data, column, feature)
    
        # Plot Analysis Results
        if unit_test == False:
            self.plot_analysis_results(normality_results, original_results, xform_results, non_parametric_results, feature)


        # Create Results Dictionary
        parametric_nonparametric_results = {
            'ttest_statistic': original_results['statistic'],
            'ttest_pvalue': original_results['p_value'],
            'cohens_d': original_results['size_effect'],
            'mwu_statistic': non_parametric_results['statistic'],
            'mwu_pvalue': non_parametric_results['p_value'],
            'effect_size_r': non_parametric_results['size_effect'],
            'normality_1': normality_results['shapiro_1'][1] > 0.05,
            'normality_2': normality_results['shapiro_2'][1] > 0.05,
            'equal_variance': original_results['levene_p'] > 0.05
        }

        # Return Results
        return parametric_nonparametric_results















# End of Page
