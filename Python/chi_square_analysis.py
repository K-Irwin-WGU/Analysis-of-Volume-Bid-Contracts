'''chi_square_analysis'''


# IMPORT PACKAGES

import numpy as np
import pandas as pd

from prettytable import PrettyTable

import math

import seaborn as sns
import matplotlib.pyplot as plt

#from scipy import stats
from scipy.stats import chi2_contingency

# Suppress Warnings
import warnings
warnings.filterwarnings('ignore')




# CREATE CHI-SQUARE ANALYSIS CLASS

class ChiSquareAnalysis():

    '''
    Class performs chi-squared analysis on the column and target
    values in the data. The method displays a heatmap and tables of the
    test results.
    
    Returns:
    --------
    chi_squared_results: dictionary

    Outputs:
    --------
    Figure
    heatmap
    tables
    '''

    # Define init Method
    def __init__(self, data=None, column=None, 
                 target=None, alpha=0.05, unit_test=False):

        '''
        Method initializes class variables.
        
        Parameters:
        -----------
        self: The ChiSquareAnalysis object
        self._data: Dataset to analyze
        self._column: Column for subgroups
        self._target: The column to cross-match against
        alpha: The alpha for measuring the chi-squared test
        unit_test: Determines if unit tests are being performed
        '''

        # Initialize Class Variables
        self._data = data
        self._column = column
        self._target = target
        self._alpha = alpha
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
    

    # Getter for Target Variable
    @property
    def target(self):  
        return self._target
    

    # Getter for Alph Variable
    @property
    def alpha(self):  
        return self._alpha
    

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
    



    # Define Setter for Target Variable
    @target.setter
    def target(self, target_column):
        if isinstance(target_column, str) and len(target_column) > 0:
            self._target = target_column
        else:
            raise ValueError('Target column must be a non-empty string.')
    



    # Define Setter for Alpha Variable
    @alpha.setter
    def alpha(self, alpha_value):
        if isinstance(alpha_value, float) and alpha_value > 0:
            self._alpha = alpha_value
        else:
            raise ValueError('Alpha value must be a non-negitive float.')
    



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
        self: The ChiSquareAnalysis object
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
        self: The ChiSquareAnalysis object
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

        table.float_format = "10.2"
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
        self: The ChiSquareAnalysis object
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
    



    ############################## Plot Methods ################################



    # Define Method for Heatmap Plot
    def heatmap_plot(self, data, column, target, ax):

        '''
        Method creates a heatmap of the column and target values.
        
        Parameters:
        -----------
        self: The ChiSquareAnalysis object
        data: The dataset to analyze
        column: Column for subgroups
        target: The column to cross-match against
        ax: The subplot figure

        Returns:
        --------
        None

        Outputs:
        --------
        heatmap plot
        '''

        # Create Title for Labels
        column_title = self.titlize_column_name(column)
        target_title = self.titlize_column_name(target)

         # Create Contingency Table
        contingency_table = pd.crosstab(data[column], data[target])

        # Create Heatmap Heatmap
        sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu', ax=ax)

        # Add Border
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.5)
            ax.spines[axis].set_color("#010101")

        # Add Title
        ax.set_title(f'Contingency Table: {column_title} vs {target_title}', fontsize=14)

        # Add Labels
        ax.set_xlabel(f'{target_title}', fontsize=12)
        ax.set_ylabel(f'{column_title}', fontsize=12)
    



    # Define Method for Figure Text
    def create_figure_text(self, data, column, target):

        '''
        Method creates text containing group totals and target percentages
        for the text box in the figure.
        
        Parameters:
        -----------
        self: The ChiSquareAnalysis object
        data: The dataset to analyze
        column: Column for subgroups
        target: The column to cross-match against

        Returns:
        --------
        figure_text: string
        '''

        # Create Contingency Table
        contingency_table = pd.crosstab(data[column], data[target])

        # Get Index Values
        indexes = contingency_table.index

        # Set Group Variables
        group_1 = indexes[0]
        group_2 = indexes[1]

        # Get Column Values
        columns = contingency_table.columns

        # Set Target 1 Variable
        target_1 = columns[0]

        # Calculate Percentages within Column Group
        row_percentages = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100

        # Calculate Group Totals
        group_1_total = contingency_table.loc[group_1].sum()
        group_2_total = contingency_table.loc[group_2].sum()

        # Calculate Group Percentages
        group_1_pct = row_percentages.loc[group_1, target_1]
        group_2_pct = row_percentages.loc[group_2, target_1]

        # Calculate Group Target Totals
        group_1_target = contingency_table.loc[group_1, target_1]
        group_2_target = contingency_table.loc[group_2, target_1]

        # Create Text for Text Figure
        figure_text = (f"Out of {group_1_total} in the {group_1}, {group_1_target} ({group_1_pct:.1f}%) have {target_1.lower()} the target\n"
                    f"Out of {group_2_total} in the {group_2}, {group_2_target} ({group_2_pct:.1f}%) have {target_1.lower()} the target")

        # Return Text
        return figure_text




    # Define Method for Plotting Data
    def plot_data(self, data, column, target):

        '''
        Method creates a figure and a heatmap of the column and target values.
        The method also adds a text box with total and percentage values.
        
        Parameters:
        -----------
        self: The ChiSquareAnalysis object
        data: The dataset to analyze
        column: Column for subgroups
        target: The column to cross-match against

        Returns:
        --------
        None

        Outputs:
        --------
        figure
        heatmap plot
        '''

        # Set Figure Style
        sns.set_theme(style='whitegrid')
        plt.rcParams['axes.facecolor'] = "#F4F3F3FF"
        plt.rcParams['font.size'] = 12

        # Create Figure
        _, (ax) = plt.subplots(1, 1, figsize=(12, 5))

        # Create Heatmap
        self.heatmap_plot(data, column, target, ax)

        # Add Border
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.5)
            ax.spines[axis].set_color("#010101")

        # Create Figure Text
        figure_text = self.create_figure_text(data, column, target)

        # Add Text Box to Figure
        plt.figtext(0.5, 0.6, figure_text, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=1))
        
        # Display Plots
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4)
        plt.show()




    ############################## Table Methods ###############################



    # Define Method to Plot Table Title
    def plot_table_title(self, title):

        '''
        Method creates an empty figure with a title.
        
        Parameters:
        -----------
        self: The ChiSquareAnalysis object
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
        Method creates a box plot for the selected data.
        
        Parameters:
        -----------
        self: The ChiSquareAnalysis object
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
        self: The ChiSquareAnalysis object
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



    # Define Method for Printing Contingency Table
    def create_table_dataframe(self, column, table):

        '''
        Method converts a table into a dataframe.
        
        Parameters:
        -----------
        self: The ChiSquareAnalysis object
        column: Column for subgroups
        table: The data table

        Returns:
        --------
        df_table_data: dataframe
        '''

        # Create Title for Headers
        column_title = self.titlize_column_name(column)

        # Get Index Values
        indexes = table.index

        # Set Group Variables
        group_1 = indexes[0]
        group_2 = indexes[1]

        # Get Column Values
        columns = table.columns

        # Set Target 1 Variable
        target_1 = columns[0]
        target_2 = columns[1]

        # Create Dataframe from Table Data
        df_table_data = pd.DataFrame({
            column_title: [
                group_1,
                group_2,
            ],
            f'{target_1}': [
                table.loc[group_1, target_1],
                table.loc[group_2, target_1]
            ],
            f'{target_2}': [
                table.loc[group_1, target_2],
                table.loc[group_2, target_2]
            ],
            'Total': [
                table.loc[group_1].sum(),
                table.loc[group_2].sum()
            ]
        })

        # Return Dataframe
        return df_table_data




    ############################## Print Methods ###############################



    # Define Method for Printing Contingency Table
    def print_contingency(self, column, table, print_table=True, plot_table=True):

        '''
        Method prints a table with the observed counts of the chi-squared analysis.
        
        Parameters:
        -----------
        self: The ChiSquareAnalysis object
        column: Column for subgroups
        table: The data table
        print_table: Prints table if True
        plot_table: Plots table if True

        Returns:
        --------
        None

        Outputs:
        --------
        print statement
        table
        '''

        # Create Dataframe
        df_table_data = self.create_table_dataframe(column, table)

        # Create Table From Table Data Dataframe
        data_table = self.create_table(df_table_data, 3)

        # Create Title Variable
        contingency_title = 'Analysis of Observed Counts'

        # Print Table
        if print_table == True:
            print()
            print(contingency_title)
            print(data_table)

        # Plot Table
        if plot_table == True:
            contingency_col_widths = [0.325, 0.225, 0.225, 0.225]
            self.plot_table_data([df_table_data], [contingency_title], contingency_col_widths, 'left')
    



    # Define Method for Printing Expected Results
    def print_expected(self, column, table, print_table=True, plot_table=True):

        '''
        Method prints a table with the expected counts of the chi-squared analysis.
        
        Parameters:
        -----------
        self: The ChiSquareAnalysis object
        column: Column for subgroups
        table: The data table
        print_table: Prints table if True
        plot_table: Plots table if True

        Returns:
        --------
        None

        Outputs:
        --------
        print statement
        table
        '''

        # Create Dataframe
        df_table_data = self.create_table_dataframe(column, table)

        # Create Table From Table Data Dataframe
        data_table = self.create_table(df_table_data, 3)

        # Create Title Variable
        expected_title = 'Analysis of Expected Counts'

        # Print Table
        if print_table == True:
            print()
            print(expected_title)
            print(data_table)

        # Plot Table
        if plot_table == True:
            expected_col_widths = [0.325, 0.225, 0.225, 0.225]
            self.plot_table_data([df_table_data], [expected_title], expected_col_widths, 'left')
    



    # Define Method for Printing Row Percentages
    def print_row_percentages(self, column, table, print_table=True, plot_table=True):

        '''
        Method prints a table with the observed row percentages of the chi-squared analysis.
        
        Parameters:
        -----------
        self: The ChiSquareAnalysis object
        column: Column for subgroups
        table: The data table
        print_table: Prints table if True
        plot_table: Plots table if True

        Returns:
        --------
        None

        Outputs:
        --------
        print statement
        table
        '''

        # Create Dataframe
        df_table_data = self.create_table_dataframe(column, table)

        # Create Table From Table Data Dataframe
        data_table = self.create_table(df_table_data, 3)

        # Create Title Variable
        percentages_title = 'Analysis of Observed Row Percentages'

        # Print Table
        if print_table == True:
            print()
            print(percentages_title)
            print(data_table)

        # Plot Table
        if plot_table == True:
            percentages_col_widths = [0.325, 0.225, 0.225, 0.225]
            self.plot_table_data([df_table_data], [percentages_title], percentages_col_widths, 'left')

    
    
    
    # Define Methond to Print Chi-Square Statistic
    def print_chi_square_statistic(self, data, column, target, expected, print_table=True, plot_table=True):

        '''
        Method prints a table of the chi-square statistic for each
        column and target.
        
        Parameters:
        -----------
        self: The ChiSquareAnalysis object
        data: The dataset to analyze
        column: Column for subgroups
        target: The column to cross-match against
        expected: The expected results of the chi-squared test
        print_table: Prints table if True
        plot_table: Plots table if True

        Returns:
        --------
        None

        Outputs:
        --------
        print statement
        table
        '''

        # Create Title for Headers
        column_title = self.titlize_column_name(column)
        target_title = self.titlize_column_name(target)

        # Create Contingency Table
        contingency_table = pd.crosstab(data[column], data[target])

        # Create Lists for Dataframe
        groups = []
        targets = []
        expected_results = []
        observed_results = []
        statistics = []


        # Create Empty NumPy Array
        chi2_components = np.zeros(shape=expected.shape)

        # Itterate Through Array
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):

                # Get Observed Value
                observed = contingency_table.iloc[i, j]

                # Get Expected Value
                exp = expected[i, j]

                # Calculate Chi-Square Statistic
                chi2_components[i, j] = ((observed - exp) ** 2) / exp
                
                # Append Results to Data Lists
                groups.append(f'{contingency_table.index[i]}')
                targets.append(f'{contingency_table.columns[j]}')
                expected_results.append(f'{exp:.1f}')
                observed_results.append(f'{observed}')
                statistics.append(f'{chi2_components[i, j]:.2f}')


        # Create Dataframe for Results Data
        df_results_data = pd.DataFrame({
            column_title: groups,
            target_title: targets,
            'Expected Results': expected_results,
            'Observed Results': observed_results,
            'Chi-Square Statistic': statistics
        })

        # Create Dataframe for the Sum of All Statistics
        df_sum_statistics = pd.DataFrame({
            'Metric': ['Sum Statistics'],
            'Result': [f'{np.sum(chi2_components):.2f}']
        })


        # Create Table Dataframes
        results_table = self.create_table(df_results_data, 3)
        sum_statistics_table = self.create_table(df_sum_statistics, 3)

        # Create Title Variables
        statistic_title = 'Analysis of Chi-Square Statistic'
        sum_statistics_title = 'Sum of All Chi-Square Statistics'

        # Print Tables
        if print_table == True:

            # Print Chi-Square Analysis
            print(f'\n{statistic_title}')
            print(results_table)

            # Print Sum Statistics
            print(f'\n{sum_statistics_title}')
            print(sum_statistics_table)
        
        # Plot Tables
        if plot_table == True:

            # Plot Chi-Square Analysis
            statistic_col_widths = [0.2, 0.2, 0.2, 0.2, 0.2]
            self.plot_table_data([df_results_data], [statistic_title], statistic_col_widths, 'left')

            # Plot Sum Statistics
            sum_statistic_col_widths = [0.5, 0.5]
            self.plot_table_data([df_sum_statistics], [sum_statistics_title], sum_statistic_col_widths, 'center')
    



    # Define Method for Printing Results
    def print_results(self, results, column, target, print_table=True, plot_table=True):

        '''
        Method prints a table of the chi-squared test results.
        
        Parameters:
        -----------
        self: The ChiSquareAnalysis object
        results: The results of the chi-squared test
        column: Column for subgroups
        target: The column to cross-match against
        print_table: Prints table if True
        plot_table: Plots table if True

        Returns:
        --------
        None

        Outputs:
        --------
        print statement
        table
        '''

        # Create Title for Headers
        column_title = self.titlize_column_name(column)
        target_title = self.titlize_column_name(target)

        # Set Results Variables
        chi2 = results['chi_square_statistic']
        p_value = results['p_value']
        alpha = results['alpha']
        dof = results['degrees_of_freedom']

        # P_Value Interpretation
        if p_value < alpha:
            p_interpretation = 'Reject H₀'
            assessment = f'There is a statistically significant association between {column_title} and {target_title}'
        else:
            p_interpretation = 'Fail to Reject H₀'
            assessment = f'There is no statistically significant association between {column_title} and {target_title}'

        # Create Dataframe for Results Data
        df_results_data = pd.DataFrame({
            'Statistic': [
                'Chi-Square Statistic',
                'Degrees of Freedom',
                'P_Value',
                'Interpretation'
            ],
            'Value': [
                f'{chi2:.2f}',
                f'{dof}',
                f'{p_value:.6f}',
                f'{p_interpretation}'
            ]
        })

        # Create Dataframe for Overall Assessment
        df_assessment = pd.DataFrame({
            'Interpretation': [assessment]
        })

        # Create Table From Stats Dataframe
        results_table = self.create_table(df_results_data, 1)
        assessment_table = self.create_table(df_assessment, 1)

        # Create Title Variables
        results_title = 'Analysis of Chi-Square Test Results'
        assessment_title = 'Overall Assessment of Chi-Square Test Analysis'

        # Print Tables
        if print_table == True:

            # Print Results Data
            print(f'\n{results_title}')
            print(results_table)

            # Print Overall Assessment
            print(f'\n{assessment_title}')
            print(assessment_table)

        # Plot Tables
        if plot_table == True:

            # Plot Results Data
            results_col_widths = [0.5, 0.5]
            self.plot_table_data([df_results_data], [results_title], results_col_widths, 'left')

            # Plot Overall Assessment
            sum_statistic_col_widths = None
            self.plot_table_data([df_assessment], [assessment_title], sum_statistic_col_widths, 'center')




    ############################## Main Method #################################



    # Define Method for Chi-Square Test Analysis
    def chi_square_test_analysis(self, column=None, target=None, alpha=None, unit_test=None):

        '''
        Method performs chi-squared analysis on the column and target
        values in the data. The method displays a heatmap and tables of the
        test results.
        
        Parameters:
        -----------
        self: The ChiSquareAnalysis object
        column: Column for subgroups
        target: The column to cross-match against
        alpha: The alpha for measuring the chi-squared test
        unit_test: Determines if unit tests are being performed

        Returns:
        --------
        chi_squared_results: dictionary

        Outputs:
        --------
        figure
        heatmap
        tables
        '''
        
        # Set Data Variable
        data = self._data

        # Set Column Variable
        if column == None:
            column = self._column
        
        # Set Target Variable
        if target == None:
            target = self._target
        
        # Set Alpha Variable
        if alpha == None:
            alpha = self._alpha
        
        # Set Unit Test Variable
        if unit_test == None:
            unit_test = self._unit_test

        # Create Contingency Table
        contingency_table = pd.crosstab(data[column], data[target])
        
        # Perform Chi-Square Test 
        chi2, p_value, dof, expected = chi2_contingency(contingency_table, correction=False)

        # Create Chi-Square Results Dictionary
        chi_squared_results = {
            'chi_square_statistic':chi2,
            'p_value': p_value,
            'alpha': alpha,
            'degrees_of_freedom': dof
        }

        # Calculate Expected Values
        expected_table = pd.DataFrame(expected,
                                    index=contingency_table.index,
                                    columns=contingency_table.columns)
        
        # Calculate Observed Percentages
        row_percentages = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100

        # Print Chi-Square Test Analysis Results
        if unit_test == False:

            # Plot Data
            self.plot_data(data, column, target)

            # Print Contingency Table
            self.print_contingency(column, contingency_table, False, True)

            # Print Expected
            self.print_expected(column, expected_table.round(2), False, True)

            # Print Row Percentages
            self.print_row_percentages(column, row_percentages.round(2), False, True)

            # Print Chi-Square Statistic
            self.print_chi_square_statistic(data, column, target, expected, False, True)

            # Print Chi-Square Results
            self.print_results(chi_squared_results, column, target, False, True)

        # Return the Chi-Square Results
        return chi_squared_results
















# End of Page
