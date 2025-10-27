'''outlier_analysis'''


# IMPORT PACKAGES

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# Suppress Warnings
import warnings
warnings.filterwarnings('ignore')




# CREATE OUTLIER ANALYSIS CLASS

class OutlierAnalysis():

    '''
    Class performs outlier analysis on data.

    Returns:
    --------
    outliers_results: dataframe/list

    Outputs:
    --------
    figure
    box plot
    histogram plot
    table
    '''

    # Define init Method
    def __init__(self, data=None, column=None, feature=None, group='all', unit_test=False):

        '''
        Method initializes class variables.
        
        Parameters:
        -----------
        self: The VisualAnalysis object
        self._data: Dataset to analyze
        self._column: Column for subgroups
        self._feature: The columns to measure
        self._group: The subgroup for the analysis
        unit_test: Determines if unit tests are being performed
        '''

        # Initialize Class Variables
        self._data = data
        self._column = column
        self._feature = feature
        self._group = group 
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
    

    # Getter for Group Variable
    @property
    def group(self):  
        return self._group
    

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
    



    # Define Setter for Group Variable
    @group.setter
    def group(self, group):
        if isinstance(group, str) and len(group) > 0:
            self._group = group
        else:
            raise ValueError('Group must be a non-empty string.')
    



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
        self: The OutlierAnalysis object
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




    ############################## Analysis Methods ############################



    # Define Method for Finding Outliers
    def find_outliers(self, data):

        '''
        Method calculates the interquartile range and returns
        the data outside that range, along with the upper and
        lower bounds.
        
        Parameters:
        -----------
        self: The OutlierAnalysis object
        data: Dataset to analyze

        Returns:
        --------
        results: list
        '''

        # Calculate First Quartile (Q1)
        q1 = data.quantile(0.25)

        # Calculate Third Quartile (Q3)
        q3 = data.quantile(0.75)

        # Calculate Interquartile Range
        iqr = q3 - q1

        # Define Outlier Boundaries
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Find Outliers
        outliers = data[(data < lower_bound) | (data > upper_bound)]

        # Create Results List
        results = [lower_bound, upper_bound, outliers]

        # Return Results List
        return results
    



    ############################## Plot Methods ################################



    # Define Method for Box Plot
    def box_plot(self, data, group, feature, ax):

        '''
        Method creates a box plot for the selected data.
        
        Parameters:
        -----------
        self: The OutlierAnalysis object
        data: Dataset to analyze
        group: The subgroup for the analysis
        feature: The columns to measure
        ax: The subplot figure

        Outputs:
        --------
        box plot
        '''
        
        # Create Boxplot
        sns.boxplot(y=data, ax=ax, color='b', legend=False)

        # Add Border
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.25)
            ax.spines[axis].set_color("#010101")

        # Add Grid
        ax.grid(True, axis = 'y', color = '#010101', linestyle = '--', linewidth = 0.5)
        
        # Create Title
        feature_title = self.titlize_column_name(feature)
        
        # Add Title and Labels
        ax.set_title(f'Box Plot of {feature_title} Showing Outliers')
        ax.set_xlabel(group)
        ax.set_ylabel(feature_title)




    # Define Method for Histogram Plot
    def histogram_plot(self, data, feature, ax):

        '''
        Method creates a histogram plot of the data along with
        the upper and lower bounds.
        
        Parameters:
        -----------
        self: The OutlierAnalysis object
        data: Dataset to analyze
        feature: The column to measure
        ax: The subplot figure

        Outputs:
        --------
        histogram
        '''

        # Create Title for Labels
        feature_title = self.titlize_column_name(feature)

        # Find Boundaries
        results = self.find_outliers(data)
        lower_bound = results[0]
        upper_bound = results[1]

        # Create Histogram
        sns.histplot(data, kde=True, ax=ax, color='blue')

        # Add Mean and Standard Deviation Lines
        ax.axvline(lower_bound, color='r', linestyle='--', label=f'Lower bound: {lower_bound:.2f}')
        ax.axvline(upper_bound, color='r', linestyle='--', label=f'Upper bound: {upper_bound:.2f}')

        # Add Border
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.25)
            ax.spines[axis].set_color("#010101")

        # Add Grid
        ax.grid(True, color = '#010101', linestyle = '--', linewidth = 0.5)

        # Add Title and Labels
        ax.set_title(f'Distribution of {feature_title} with Outlier Boundaries')
        ax.set_xlabel(feature_title)
        ax.set_ylabel('Frequency')
        
        # Add Legend
        legend = ax.legend(shadow=True)
        legend.get_frame().set_edgecolor("#010101")




    # Define Method to Plot Outliers
    def plot_outliers(self, data, group, feature):

        '''
        Method creates a figure displaying a box plot
        and a histogram for visual analysis of data.
        
        Parameters:
        -----------
        self: The OutlierAnalysis object
        data: Dataset to analyze
        group: The subgroup for the analysis
        feature: The column to measure

        Outputs:
        --------
        figure
        box plot
        histogram
        '''

        # Print Empty Line for Spacing
        print()

        # Set Figure Style
        sns.set_theme(style='whitegrid')
        plt.rcParams['axes.facecolor'] = "#F4F3F3FF"
        plt.rcParams['font.size'] = 12

        # Create Figure
        _, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Create Box Plot of Data Value
        ax1 = axes[0]
        self.box_plot(data, group, feature, ax1)

        # Create Histogram of Data Values
        ax2 = axes[1]
        self.histogram_plot(data, feature, ax2)

        # Create Title for Labels
        feature_title = self.titlize_column_name(feature)

        # Display Plots
        plt.tight_layout()
        plt.suptitle(f'{group} Outlier Analysis for {feature_title}', fontsize=16, y=1.02)
        plt.show()




    ############################## Table Methods ###############################



    # Define Method for Plot Table
    def table_plot(self, dataframe, group, feature, ax):

        '''
        Method creates a table plot for the selected data.
        
        Parameters:
        -----------
        self: The OutlierAnalysis object
        dataframe: Dataset to plot
        group: The subgroup of the analysis
        feature: The columns to measure
        ax: The subplot figure

        Outputs:
        --------
        table
        '''

        # Create Title for Labels
        feature_title = self.titlize_column_name(feature)

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
                        cellColours=cell_colors,
                        colLabels=column_labels,
                        colColours=column_colors,
                        loc='upper center')
        
        # Iterate through Column Header and Set Text Color
        for i in range(len(column_labels)):
            cell = table[0, i]
            cell.get_text().set_color('#ebeced')

        # Set Table Properties
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        
        # Add Title
        ax.set_title(f'\nSummary of Outlier Analysis for {feature_title} in {group}', fontsize=12)
    



    # Define Method to Plot Table Data
    def plot_table_data(self, data, group, feature):

        '''
        Method creates a figure and a table plot.
        
        Parameters:
        -----------
        self: The OutlierAnalysis object
        data: Dataset to plot
        group: The subgroup of the analysis
        feature: The column of measure

        Outputs:
        --------
        figure
        table
        '''

        # Set Figure Style
        sns.set_style(style='whitegrid')

        # Create Subplots
        _, ax = plt.subplots(1, 1, figsize=(5, 2))

        # Plot Table
        self.table_plot(data, group, feature, ax)
        
        # Display Table
        plt.show()




    ############################## Run Methods #################################



    # Define Method for Outlier Analysis
    def outlier_analysis(self, data, group, feature, unit_test):
        
        '''
        Method performs outlier analysis of data, including visual
        and statistical analysis.
        
        Parameters:
        -----------
        self: The OutlierAnalysis object
        data: Dataset to analyze
        group: The subgroup for the analysis
        feature: The column to measure
        unit_test: Determines if unit tests are being performed

        Returns:
        --------
        outliers: dataset

        Outputs:
        --------
        graph
        table
        '''
        
        # Set Data Values
        values = data[feature]

        # Find Outliers
        lower_bound, upper_bound, outliers = self.find_outliers(values)

        # Create Dataframe for Results Data
        df_results_data = pd.DataFrame({
            'Statistic': [
                'Number of Outliers',
                'Percentage of Outliers',
                'Upper Bound',
                'Lower Bound',
                'Confidence Interval'
            ],
            'Value': [
                f'{len(outliers)}',
                f'{(len(outliers) / len(values)) * 100:.2f}',
                f'{upper_bound:.2f}',
                f'{lower_bound:.2f}',
                f'{(upper_bound - lower_bound):.2f}'
            ]
        })

        # Plot Outliers
        if unit_test == False:
            self.plot_outliers(values, group, feature)

        # Print Results Table
        if unit_test == False:
            self.plot_table_data(df_results_data, group, feature)

        # Return the outliers
        return outliers




    ############################## Main Methods ################################



    # Define Method to Apply Outlier Analysis
    def apply_outlier_analysis(self, column=None, feature=None, group=None, unit_test=None):

        '''
        Method performs outlier analysis on all groups in the selected column.
        
        Parameters:
        -----------
        self: The OutlierAnalysis object
        column: Column for subgroups
        feature: The columns to measure
        group: The subgroup for the analysis
        unit_test: Determines if unit tests are being performed

        Returns:
        --------
        outliers_results: dataset/list

        Outputs:
        --------
        graph
        table
        '''

        # Set Data Variable
        data = self._data

        # Set Column Variable
        if column == None:
            column = self._column

        # Set Feature Variable
        if feature == None:
            feature = self._feature
        
        # Set Group Variable
        if group == None:
            group = self._group
        
        # Set Unit Test Variable
        if unit_test == None:
            unit_test = self._unit_test
        
        # Create Group List
        group_list = data[column].value_counts().head(8).keys()

        # Set Group Variable
        group_1 = group_list[0]
        group_2 = group_list[1]

        if group == 'first':

            # Filter Data
            filtered_data = data[data.apply(lambda row: row[column] == group_1, axis=1)]

            # Perform Outlier Analysis on Filtered Data
            outlier_results = self.outlier_analysis(filtered_data, group_1, feature, unit_test)
        
        elif group == 'second':

            # Filter Data
            filtered_data = data[data.apply(lambda row: row[column] == group_2, axis=1)]

            # Perform Outlier Analysis on Filtered Data
            outlier_results = self.outlier_analysis(filtered_data, group_2, feature, unit_test)
        
        else:

            # Create Outlier List
            outlier_results = []

            # Iterate Through Groups in Group List
            for group in group_list:

                # Filter Data by Group
                filtered_data = data[data.apply(lambda row: row[column] == group, axis=1)]

                # Apply Outlier Analysis on Filtered Data
                outliers = self.outlier_analysis(filtered_data, group, feature, unit_test)

                # Append Outliers to List
                outlier_results.append(outliers)
        
        # Return Outlier Results
        return outlier_results
















# End of Page
