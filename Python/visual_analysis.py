'''visual_analysis'''


# IMPORT PACKAGES

import numpy as np
import pandas as pd

import math

import seaborn as sns
import matplotlib.pyplot as plt

# Suppress Warnings
import warnings
warnings.filterwarnings('ignore')




# CREATE VISUAL ANALYSIS CLASS

class VisualAnalysis():

    '''
    Class performs visual and statistical analysis on data.

    Returns:
    --------
    None

    Outputs:
    --------
    figure
    histogram plots
    box plots
    count plots
    violin plots
    kde plots
    tables
    '''

    # Define init Method
    def __init__(self, data=None, column=None, features=None, feature_1=None, 
                 feature_2=None, target=None, target_slice=None,  unit_test=False):
        
        '''
        Method initializes class variables.
        
        Parameters:
        -----------
        self: The VisualAnalysis object
        self._data: Dataset to analyze
        self._column: Column for subgroups
        self._features: list of columns to measure
        self._feature_1: The first column to measure
        self._feature_2: The second column to measure
        self._target: Column of the target of comparison
        target_slice: Number of charectors to include in target value
        unit_test: Determines if unit tests are being performed
        '''

        # Initialize Class Variables
        self._data = data
        self._column = column
        self._features = features
        self._feature_1 = feature_1
        self._feature_2 = feature_2
        self._target = target
        self._target_slice = target_slice
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
    

    # Getter for Features Variable
    @property
    def features(self):  
        return self._features
    

    # Getter for Feature 1 Variable
    @property
    def feature_1(self):  
        return self._feature_1
    

    # Getter for Feature 2 Variable
    @property
    def feature_2(self):  
        return self._feature_2
    

    # Getter for Target Variable
    @property
    def target(self):  
        return self._target
    

    # Getter for Target Slice Variable
    @property
    def target_slice(self):  
        return self._target_slice
    

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
    



    # Define Setter for Features Variable
    @features.setter
    def features(self, data_list):
        if isinstance(data_list, list) and len(data_list) > 0:
            self._features = data_list
        else:
            raise ValueError('Data list must be a non-empty list.')
    



    # Define Setter for Feature 1 Variable
    @feature_1.setter
    def feature_1(self, feature_column):
        if isinstance(feature_column, str) and len(feature_column) > 0:
            self._feature_1 = feature_column
        else:
            raise ValueError('Feature column must be a non-empty string.')
    



    # Define Setter for Feature 2 Variable
    @feature_2.setter
    def feature_2(self, feature_column):
        if isinstance(feature_column, str) and len(feature_column) > 0:
            self._feature_2 = feature_column
        else:
            raise ValueError('Feature column must be a non-empty string.')
    



    # Define Setter for Target Variable
    @target.setter
    def target(self, target_column):
        if isinstance(target_column, str) and len(target_column) > 0:
            self._target = target_column
        else:
            raise ValueError('Target column must be a non-empty string.')
    



    # Define Setter for Target Slice Variable
    @target_slice.setter
    def target_slice(self, slice_value):
        if isinstance(slice_value, int) and slice_value > 0:
            self._target_slice = slice_value
        elif slice_value == None:
            self._target_slice = slice_value
        else:
            raise ValueError('Slice value must be a positive integer or None.')
    



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
        self: The VisualAnalysis object
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




    # Define Method For Percentage in Count Plot
    def count_plot_percentage(self, ax):

        '''
        Method adds percentages to the bars in a count plot.
        
        Parameters:
        -----------
        self: The VisualAnalysis object
        ax: The figure container for the count plot 

        Outputs:
        --------
        bar percentages
        '''

        # Calculate Bar Heights
        all_heights = [[p.get_height() for p in bars] for bars in ax.containers]
        
        # Iterate Over Bars in Figure Container
        for bars in ax.containers:
            for i, p in enumerate(bars):

                # Calculate Total for All Bar Hights
                total = sum(xgroup[i] for xgroup in all_heights)

                # Calculate Percentage of Total
                percentage = f'{(100 * p.get_height() / total) :.1f}%'

                # Add Percentage to Bar in Count Plot
                ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height()), size=11, ha='center', va='bottom')
    



    ############################## Plot Methods ################################



    # Define Histogram Method
    def histogram(self, data, group, feature, ax):

        '''
        Method creates a histogram displaying the selected data.
        
        Parameters:
        -----------
        self: The VisualAnalysis object
        data: Dataset to analyze
        group: The subgroup from the subgroup column
        feature: The column to measure
        ax: The subplot figure

        Outputs:
        --------
        histogram plot
        '''

        # Calculate Measures of Central Tendency
        mean_val = data.mean()
        median_val = data.median()
        mode_val = data.mode()[0]

        
        # Create Histogram Plot
        sns.histplot(data, kde=True, color='blue')
        
        # Add Visual Representation of Mean, Median, and Mode
        plt.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
        plt.axvline(median_val, color='g', linestyle='-.', label=f'Median: {median_val:.2f}')
        plt.axvline(mode_val, color='b', linestyle=':', label=f'Mode: {mode_val:.2f}')

        # Add Border
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.25)
            ax.spines[axis].set_color("#010101")
        
        # Add Grid
        plt.grid(True, color = '#010101', linestyle = '--', linewidth = 0.5)

        # Create Plot Title
        feature_title = self.titlize_column_name(feature)
        title = f'Distribution of {group} by {feature_title}'

        # Add Title
        plt.title(title)

        # Add Legend
        legend = plt.legend(shadow=True)
        legend.get_frame().set_edgecolor("#010101")




    # Define Box Plot Method
    def box_plot(self, data, column, feature, ax):

        '''
        Method creates a box plot for the selected column and feature.
        
        Parameters:
        -----------
        self: The VisualAnalysis object
        data: Dataset to analyze
        target: Column for subgroups
        feature: The columns to measure
        ax: The subplot figure

        Outputs:
        --------
        box plot
        '''

        # Select Key Column Values for Pallette
        group_list = data[column].value_counts().head(2).keys()
        group_1 = group_list[0]
        group_2 = group_list[1]

        # Create Color Palette
        palette={group_1: '#4361EE', group_2: '#3A0CA3'}
        #palette={group_1: 'red', group_2: 'blue'}
        
        # Create Boxplot
        sns.boxplot(x=column, hue=column, y=feature, data=data, palette=palette, legend=False)

        # Add Border
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.25)
            ax.spines[axis].set_color("#010101")
        
        # Add Grid
        plt.grid(True, axis = 'y', color = '#010101', linestyle = '--', linewidth = 0.5)

        # Create Title
        feature_title = self.titlize_column_name(feature)
        target_title = self.titlize_column_name(column)
        
        # Add Title and Labels
        plt.title(f'Distribution of {feature_title}')
        plt.xlabel(target_title)
        plt.ylabel(feature_title)




    # Define Violin Plot Method
    def violin_plot(self, data, column, feature, value, ax):

        '''
        Method creates a violin plot for the selected column and feature.
        
        Parameters:
        -----------
        self: The VisualAnalysis object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The column to measure
        value: The value of the target column
        ax: The subplot figure

        Outputs:
        --------
        violin plot
        '''
            
        # Create Titles for Visuals
        column_title = self.titlize_column_name(column)
        feature_title = self.titlize_column_name(feature)

        # Select Key Column Values for Palette
        group_list = data[column].value_counts().head(2).keys()
        group_1 = group_list[0]
        group_2 = group_list[1]

        # Create Color Palette
        palette={group_1: '#4361EE', group_2: '#3A0CA3'}
        
        # Create Violin Plot
        sns.violinplot(x=column, hue=column, y=feature, data=data, palette=palette, legend=False)

        # Add Border
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.25)
            ax.spines[axis].set_color("#010101")
        
        # Add Grid
        plt.grid(axis = 'y', color = '#010101', linestyle = '--', linewidth = 0.5)
        
        # Add Title and Labels
        plt.title(f'Distribution of {value}')
        plt.xlabel(column_title)
        plt.ylabel(feature_title)




    # Define KDE Plot Method
    def kde_plot(self, data, group, feature, target, ax):

        '''
        Method creates a KDE plot comparing subgroups against the target value.
        
        Parameters:
        -----------
        self: The VisualAnalysis object
        data: Dataset to analyze
        group: The subgroup from the subgroup column
        feature: The column to measure
        target: Column of the target of comparison
        ax: The subplot figure

        Outputs:
        --------
        KDE plot
        '''

        # Create Lists of Target Values
        target_values = data[target].unique()    
        target_1 = target_values[0]
        target_2 = target_values[1]

        # Separate Data by Target
        group1 = data[data[target] == target_1][feature]
        group2 = data[data[target] == target_2][feature]
        
        # Plot Actual Distributions
        sns.kdeplot(group1, ax=ax, fill=True, color='blue', label=f'{target_1}')
        sns.kdeplot(group2, ax=ax, fill=True, color='green', label=f'{target_2}')

        # Add Border
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.25)
            ax.spines[axis].set_color("#010101")

        # Add Grid
        ax.grid(True, color = '#010101', linestyle = '--', linewidth = 0.5, alpha=0.75)

        # Add Vertical lines for Means
        mean1, mean2 = group1.mean(), group2.mean()
        ax.axvline(mean1, color='blue', linestyle='--',
                    label=f'{target_1} Mean: {mean1:.2f}')
        ax.axvline(mean2, color='green', linestyle='--',
                    label=f'{target_2} Mean: {mean2:.2f}')

        # Add Annotation for Mean Difference
        mean_diff = mean1 - mean2
        ax.annotate(f'Mean Difference: {mean_diff:.2f}',
                    xy=((mean1 + mean2) / 2, 0.02),
                    xytext=((mean1 + mean2) / 2, 0.04),
                    ha='center',
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1))

        # Create Title for Labels
        feature_title = self.titlize_column_name(feature)
        
        # Set Labels
        ax.set_title(f'{group} Distributions by {feature_title}', fontsize=14)
        ax.set_xlabel(f'{feature_title}', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        #ax.grid(True, alpha=0.3)

        # Add Legend
        legend = ax.legend(shadow=True)
        legend.get_frame().set_edgecolor("#010101")
    



    ############################## Table Methods ###############################



    # Define Method for Plot Table
    def table_plot(self, dataframe, title, ax, col_widths=None, cell_location='right'):

        '''
        Method creates a table plot for the selected data.
        
        Parameters:
        -----------
        self: The VisualAnalysis object
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
        
        # Iterate through Column Header and Set Text Color
        for i in range(len(column_labels)):
            cell = table[0, i]
            cell.get_text().set_color('#ebeced')

        # Set Table Properties
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        
        # Add Title
        ax.set_title(title, fontsize=12)
    



    # Define Method to Get Column Widths
    def get_column_widths(self, data):

        '''
        Method creates summary statistics for each subgroup and each target value.
        
        Parameters:
        -----------
        self: The VisualAnalysis object
        data: Dataframe used for table

        Returns:
        --------
        col_widths: list
        '''
        
        # Set Variables
        col_widths = []
        total_width = 0
        x = 0

        # Perform While Loop to Get Width Values
        while x < data.shape[1]:

            # Calculate Width Value
            width_value = round(8 / (data.shape[1] - 1), 2)
            width_value = int(width_value * 10) /100

            # Add Width Value to Total
            total_width += width_value

            # Append Width Value to Column Widths
            col_widths.append(width_value)

            # Increment X
            x += 1

        # Prepend Lead Width to Column Widths
        lead_width = round(1 - total_width, 2)
        col_widths.insert(0, lead_width)

        # Return Column Widths
        return col_widths
    



    # Define Method to Plot Table Data
    def plot_table_data(self, data, title, col_widths=None, cell_location='right'):

        '''
        Method creates a figure and a table plot.
        
        Parameters:
        -----------
        self: The VisualAnalysis object
        data: the dataset to plot
        title: The title of the plot
        col_widths: List of column widths
        cell_location: The test alignment

        Outputs:
        --------
        figure
        table
        '''

        # Set Figure Style
        sns.set_style(style='whitegrid')

        # Create Subplots
        _, ax = plt.subplots(1, 1, figsize=(12, 1))

        # Plot Table
        self.table_plot(data, title, ax, col_widths, cell_location)
        
        # Display Table
        plt.show()




    ############################## Analysis Methods ############################



    # Define Method for Histogram Stats
    def histogram_plot_stats(self, data, column, features):

        # Create List of Column Values
        column_values = data[column].unique()

        '''
        Method creates a dataframe of histogram plot statistics
        by calculating measures of central tendency.
        
        Parameters:
        -----------
        self: The VisualAnalysis object
        data: Dataset to analyze
        column: Column for subgroups
        features: list of columns to measure

        Returns:
        --------
        df_stats: dictionary
        '''

        # Create Title for Labels
        column_title = self.titlize_column_name(column)

        # Create Lists
        group_vals = []
        feature_vals = []
        mean_vals = []
        median_vals = []
        mode_vals = []
        
        # Iterate Over Column Values
        for column_value in column_values:

            # Filter Data by Column Value
            filtered_data = data[data.apply(lambda row: row[column] == column_value, axis=1)]

            # Iterate Over Features
            for feature in features:

                # Create Title for Labels
                feature_title = self.titlize_column_name(feature)

                # Append Lists
                group_vals.append(column_value)
                feature_vals.append(feature_title)

                # Calculate Measures of Central Tendency
                mean_vals.append(filtered_data[feature].mean())
                median_vals.append(filtered_data[feature].median())
                mode_vals.append(filtered_data[feature].mode()[0])

        # Create Stats Dictionary
        stats_dict = {
            f'{column_title}': group_vals,
            'Feature': feature_vals,
            'Mean': mean_vals,
            'Median': median_vals,
            'Mode': mode_vals
        }

        # Create Stats Dataframe
        df_stats = pd.DataFrame(stats_dict)

        # Round Stats
        df_stats = df_stats.round(2)

        # Return Stats Dataframe
        return df_stats




    # Define Method For Analyzing Data with Histogram Plots
    def histogram_plot_analysis(self, column=None, features=None):

        '''
        Method creates a series of histograms displaying measures of central tendency
        for the selected column and features.
        
        Parameters:
        -----------
        self: The VisualAnalysis object
        data: Dataset to analyze
        column: Column for subgroups
        features: list of columns to measure

        Returns:
        --------
        None

        Outputs:
        --------
        figure
        histogram plots
        '''

        # Set Data Variable
        data = self._data

        # Set Column Variable
        if column == None:
            column = self._column

        # Set Features Variable
        if features == None:
            features = self._features
        
        # Select Key Column Values for Analysis
        group_list = data[column].value_counts().head(8).keys()
        
        # Calculate Number of Columns
        if len(group_list) > 4:
            c = 3
        else:
            c = 2

        # Calculate Number of Rows
        r = math.ceil(len(group_list) / c) + 1
        i = 1
        
        # Set Figure Style
        sns.set_theme(style='whitegrid')
        plt.rcParams['axes.facecolor'] = "#F4F3F3FF"
        plt.rcParams['font.size'] = 12

        # Create Figure
        fig = plt.figure(figsize=(15, 10))
        
        # Iterate Over Groups and Features
        for group in (group_list):
            for feature in (features):
                
                # Filter Data
                filtered_data = data[data.apply(lambda row: row[column] == group, axis=1)]
                values = filtered_data[feature]
                
                # Create Subplot
                ax = fig.add_subplot(r, c, i)
                i += 1

                # Create Histogram
                self.histogram(values, group, feature, ax)

        # Display Plots
        plt.tight_layout()
        plt.suptitle('Measure of Central Tendency for Selected Features', y=1.02, fontsize=16)
        plt.show()

        # Create Title for Labels
        column_title = self.titlize_column_name(column)

        # Get Plot Statistics
        df_stats = self.histogram_plot_stats(data, column, features)
     
        # Plot Group Statistics
        stats_title = f'\nMeasure of Central Tendency for Selected Features in Each {column_title}'
        col_widths = None
        self.plot_table_data(df_stats, stats_title, col_widths, 'left')
    



    # Define Method for Box Plot Stats
    def box_plot_stats(self, data, column, features):

        '''
        Method creates a dataframe of box plot statistics
        by calculating quartile values.
        
        Parameters:
        -----------
        self: The VisualAnalysis object
        data: Dataset to analyze
        column: Column for subgroups
        features: list of columns to measure

        Returns:
        --------
        df_quartile: dictionary
        '''

        # Create Title for Labels
        column_title = self.titlize_column_name(column)

        # Create List of Column Values
        column_values = data[column].unique()

        # Create Lists
        group_vals = []
        feature_vals = []
        q1_vals = []
        q2_vals = []
        q3_vals = []
        
        # Iterate Over Column Values
        for column_value in column_values:

            # Filter Data by Column Value
            filtered_data = data[data.apply(lambda row: row[column] == column_value, axis=1)]

            # Iterate Over Features
            for feature in features:

                # Create Title for Labels
                feature_title = self.titlize_column_name(feature)

                # Append Lists
                group_vals.append(column_value)
                feature_vals.append(feature_title)

                # Calculate Quartile Values
                q1_vals.append(filtered_data[feature].quantile(0.25))
                q2_vals.append(filtered_data[feature].quantile(0.50))
                q3_vals.append(filtered_data[feature].quantile(0.75))

        # Create Quartile Dictionary
        quartile_dict = {
            f'{column_title}': group_vals,
            'Feature': feature_vals,
            'First Quartile': q1_vals,
            'Second Quartile': q2_vals,
            'Third Quartile': q3_vals
        }

        # Create Quartile Dataframe
        df_quartile = pd.DataFrame(quartile_dict)

        # Round Values
        df_quartile = df_quartile.round(2)

        # Sort by Feature
        df_quartile = df_quartile.sort_values(by='Feature', ascending=False)

        # Return Quartile Dataframe
        return df_quartile




    # Define Method For Analyzing Data with Box Plots
    def box_plot_analysis(self, column=None, features=None):

        '''
        Method creates a series of box plots displaying measures of central tendency
        and outliers for the selected column and features.

        Parameters:
        -----------
        self: The VisualAnalysis object
        data: Dataset to analyze
        column: Column for subgroups
        features: list of columns to measure

        Returns:
        --------
        None

        Outputs:
        --------
        figure
        box plots
        '''

        # Set Data Variable
        data = self._data

        # Set Column Variable
        if column == None:
            column = self._column

        # Set Features Variable
        if features == None:
            features = self._features
        
        # Calculate Number of Columns
        if len(features) > 4:
            c = 3
        else:
            c = 2

        # Calculate Number of Rows
        r = math.ceil(len(features) / c) + 1

        # Set Figure Style
        sns.set_theme(style='whitegrid')
        plt.rcParams['axes.facecolor'] = "#F4F3F3FF"
        plt.rcParams['font.size'] = 12
        
        # Create Figure
        fig = plt.figure(figsize=(15, 10))
        
        # Iterate Over Features 
        for i, feature in enumerate(features, 1):

            # Create Subplot
            ax = fig.add_subplot(r, c, i)

            # Create Box Plot
            self.box_plot(data, column, feature, ax)
        
        # Display Pots
        plt.tight_layout()
        plt.suptitle('Box Plots with Data Points for Selected Features', y=1.02, fontsize=16)
        plt.show()

        # Get Quartile Values
        df_quartile = self.box_plot_stats(data, column, features)

        # Create Title for Labels
        column_title = self.titlize_column_name(column)
     
        # Plot Quartile Values
        quartile_title = f'\nQuartile Values for Selected Features in Each {column_title}'
        col_widths = None
        self.plot_table_data(df_quartile, quartile_title, col_widths, 'left')
    



    def cout_plot_stats(self, data, column, target, target_slice):

        '''
        Method creates a dataframe of count plot statistics.
        
        Parameters:
        -----------
        self: The VisualAnalysis object
        data: Dataset to analyze
        column: Column for subgroups
        target: Column of the target of comparison

        Returns:
        --------
        df_stats: dictionary
        '''

        # Create List of Column Values
        column_values = data[column].unique()

        # Create List of Target Values
        target_values = data[target].unique()

        # Create Title for Labels
        column_title = self.titlize_column_name(column)

        # Create Stats Dictionary
        stats_dict = {
            f'{column_title}': column_values
        }

        # Iterate Over Target Values
        for target_value in target_values:

            # Filter Data
            filtered_data = data[data.apply(lambda row: row[target] == target_value, axis=1)]

            # Create Counts List
            counts = []
            
            # Iterate Over Column Values
            for column_value in column_values:

                # Calculate Total 
                count = filtered_data[filtered_data.apply(lambda row: row[column] == column_value, axis=1)].shape[0]

                # Append to Counts List
                counts.append(count)

            # Create Column Name
            if target_slice == None:
                column_name = target_value
            else:
                column_name = target_value[:target_slice]

            # Append to Stats Dictionary
            stats_dict[f'{column_name}'] = counts

        # Create Stats Dataframe
        df_stats = pd.DataFrame(stats_dict)

        # Return Stats Dataframe
        return df_stats




    # Define Method For Analyzing Data with Count Plot
    def count_plot_analysis(self, column=None, target=None, target_slice=None):

        '''
        Method creates a count plot of subgroups comparing a target status.
        
        Parameters:
        -----------
        self: The VisualAnalysis object
        data: Dataset to analyze
        column: Column for subgroups
        target: Column of the target of comparison
        target_slice: Number of charectors to include in target value

        Returns:
        --------
        None

        Outputs:
        --------
        figure
        column plot
        '''

        # Set Data Variable
        data = self._data

        # Set Column Variable
        if column == None:
            column = self._column

        # Set Target Variable
        if target == None:
            target = self._target
        
        # Set Target Slice Variable
        if target_slice == None:
            target_slice = self._target_slice
        
        # Create List of Target Values
        target_values = data[target].unique()    
        target_1 = target_values[0]
        target_2 = target_values[1]

        # Create List for Labels
        label_list = []
        for value in target_values:
            label_list.append(f'{value}')

        # Create Title for Labels
        column_title = self.titlize_column_name(column)
        target_title = self.titlize_column_name(target)

        # Set Figure Style
        sns.set_theme(style='whitegrid')
        plt.rcParams['axes.facecolor'] = "#F4F3F3FF"
        plt.rcParams['font.size'] = 12

        # Create Figure
        plt.figure(figsize=(15, 6))

        # Create Count Plot
        if len(target_values) == 2:
            ax = sns.countplot(x=column, hue=target, data=data, palette={target_1: 'blue', target_2: 'green'})
        else:
            palette = "viridis"
            sns.set_palette(palette)
            ax = sns.countplot(x=column, hue=target, data=data, palette=palette)

        # Add Border
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.25)
            ax.spines[axis].set_color("#010101")
        
        # Add Grid
        plt.grid(axis = 'y', color = '#010101', linestyle = '--', linewidth = 0.5)
            
        # Add Percetages
        self.count_plot_percentage(ax)

        # Add Labels
        ax.set_title(f'Count of {target_title} by {column_title}')
        ax.set_xlabel(f'{column_title}')
        ax.set_ylabel('Count')

        # Add Legend
        legend = ax.legend(title=f'{target_title}', labels=label_list, shadow=True)
        legend.get_frame().set_edgecolor("#010101")
        plt.show()

        # Get Plot Statistics
        df_stats = self.cout_plot_stats(data, column, target, target_slice)

        # Set Column Widths
        if target_slice == None:
            col_widths = None
        else:
            col_widths = self.get_column_widths(df_stats)
     
        # Plot Group Statistics
        stats_title = f'\nTotal Number of {target_title} Values for Each {column_title}'
        self.plot_table_data(df_stats, stats_title, col_widths, 'left')
    



    # Define Method for Violin Plot Stats
    def violin_plot_stats(self, data, column, feature, target):

        '''
        Method creates a dataframe of violin plot statistics
        by calculating the median, interquartile range, and range.
        
        Parameters:
        -----------
        self: The VisualAnalysis object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The column to measure
        target: Column of the target of comparison

        Returns:
        --------
        df_stats: dictionary
        '''

        # Create Lists
        group_vals = []
        target_vals = []
        median_vals = []
        iqr_vals = []
        range_vals = []

        # Create List of Column Values
        column_values = data[column].unique()

        # Select Key Target Values for Analysis
        target_list = data[target].value_counts().head(9).keys()
        target_list = sorted(target_list)
        
        # Iterate Over Target List
        for target_value in target_list:

            # Filter Data by Target Value
            filtered_data = data[data.apply(lambda row: row[target] == target_value, axis=1)]

            # Iterate Over Column Values
            for column_value in column_values:

                # Append Lists
                group_vals.append(column_value)
                target_vals.append(target_value)

                # Filter Data by Column Value
                sub_filtered_data = filtered_data[filtered_data.apply(lambda row: row[column] == column_value, axis=1)]

                # Append Median Value
                median_vals.append(sub_filtered_data[feature].median())
                
                # Calculate Interquartile Range
                q1 = sub_filtered_data[feature].quantile(0.25)
                q3 = sub_filtered_data[feature].quantile(0.75)
                iqr_vals.append(q3 - q1)

                # Calculate Range
                max_value = sub_filtered_data[feature].max()
                min_value = sub_filtered_data[feature].min()
                range_vals.append(max_value - min_value)
        
        # Create Title for Labels
        column_title = self.titlize_column_name(column)
        target_title = self.titlize_column_name(target)

        # Create Stats Dictionary
        stats_dict = {
            f'{target_title}': target_vals,
            f'{column_title}': group_vals,
            'Median': median_vals,
            'Interquartile Range': iqr_vals,
            'Range': range_vals
        }

        # Create Stats Dataframe
        df_stats = pd.DataFrame(stats_dict)

        # Round Stats
        df_stats = df_stats.round(2)

        # Return Stats Dataframe
        return df_stats




    # Define Method For Analyzing Data with Violin Plots
    def violin_plot_analysis(self, column=None, feature=None, target=None):

        '''
        Method creates violin plots comparing subgroups for each target value.
        
        Parameters:
        -----------
        self: The VisualAnalysis object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The column to measure
        target: Column of the target of comparison

        Returns:
        --------
        None

        Outputs:
        --------
        figure
        violin plots
        '''

        # Set Data Variables
        data = self._data

        # Set Column Variable
        if column == None:
            column = self._column
        
        # Set Feature Variable
        if feature == None:
            feature = self._feature_1

        # Set Target Variable
        if target == None:
            target = self._target

        # Create Title for Labels
        column_title = self.titlize_column_name(column)
        feature_title = self.titlize_column_name(feature)
        target_title = self.titlize_column_name(target)

        # Select Key Target Values for Analysis
        value_list = data[target].value_counts().head(9).keys()
        value_list = sorted(value_list)

        # Calculate Number of Columns
        if len(value_list) > 4:
            c = 3
        else:
            c = 2

        # Calculate Number of Rows
        r = math.ceil(len(value_list) / c)

        # Set Figure Style
        sns.set_theme(style='whitegrid')
        plt.rcParams['axes.facecolor'] = "#F4F3F3FF"
        plt.rcParams['font.size'] = 12
        
        # Create Figure
        fig = plt.figure(figsize=(15, 10))

        # Iterate Over Values in Values List
        for i, value in enumerate(value_list, 1):
            
            # Filter Data by Target Equal to Value
            filtered_data = data[data.apply(lambda row: row[target] == value, axis=1)]
            
            # Create Subplot
            ax = fig.add_subplot(r, c, i)

            # Create Violin Plot
            self.violin_plot(filtered_data, column, feature, value, ax)

        # Display Pots
        plt.tight_layout()
        plt.suptitle(f'Violin Plots for {feature_title} by {target_title}', y=1.02, fontsize=16)
        plt.show()

        # Get Violin Plot Statistics
        df_stats = self.violin_plot_stats(data, column, feature, target)
     
        # Plot Violin Statistics
        stats_title = f'\nViolin Plot Details for Each {target_title} by Each {column_title}'
        col_widths = None
        self.plot_table_data(df_stats, stats_title, col_widths, 'left')




    # Define a Method to Visualize Data with KDE Plots
    def kde_plot_analysis(self, column=None, features=None, target=None):

        '''
        Method visualizes main data groups and targets by key features
        and displays differences in means.

        Parameters:
        -----------
        self: The VisualAnalysis object
        data: Dataset to be analyzed
        column: The column by which the data is segmented
        features: The list of columns by which the values are measured
        target: The column by which the segments are compared

        Returns:
        --------
        None

        Outputs:
        --------
        figure
        kde plots
        '''

        # Set Data Variables
        data = self._data

        # Set Column Variable
        if column == None:
            column = self._column
        
        # Set Features Variable
        if features == None:
            features = self._features

        # Set Target Variable
        if target == None:
            target = self._target
        
        # Create Lists from Parameter Values
        group_values = data[column].unique()
        group_values = sorted(group_values)

        # Set Figure Style
        sns.set_theme(style='whitegrid')
        plt.rcParams['axes.facecolor'] = "#F4F3F3FF"
        plt.rcParams['font.size'] = 12
            
        # Create a Figure with 4 Subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 15))
        ax =[ax1, ax2, ax3, ax4]
        i = 0
        
        # Iterate Through Group Values
        for group in group_values:
            
            # Filter Data by Group
            filtered_data = data[data.apply(lambda row: row[column] == group, axis=1)]
            
            # Iterate Through Measures
            for feature in features:

                # Create KDE Plot
                self.kde_plot(filtered_data, group, feature, target, ax[i])
                
                # Increment i
                i += 1
        
        # Display Plots
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        plt.show()



    
    ############################## Statistic Methods ###########################



    # Define Method for Summary Stats
    def get_summary_stats(self, data, column, feature, target):

        '''
        Method creates summary statistics for each subgroup and each target value.
        
        Parameters:
        -----------
        self: The VisualAnalysis object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The column to measure
        target: Column of the target of comparison

        Returns:
        --------
        df_target_1: dataframe
        df_target_2: dataframe
        '''

        # Create Title for Labels
        column_title = self.titlize_column_name(column)
        
        # Create Summary Dictionary
        summary_dict = {}

        # Create Data Lists
        group_list = data[column].value_counts().head(2).keys()
        target_list = data[target].value_counts().head(2).keys()

        # Set Target Variables
        target_1 = target_list[0]
        target_2 = target_list[1]
        

        # Iterate Through Groups in Group List
        for group in group_list:

            # Filter Data by Group
            filtered_data = data[data.apply(lambda row: row[column] == group, axis=1)]
            
            # Calculate Stats for First Target
            target_one_stats = filtered_data[filtered_data[target] == target_1][feature].agg(['mean', 'median', 'std', 'min', 'max']).to_dict()
            
            # Calculate Stats for Second Target
            target_two_stats = filtered_data[filtered_data[target] == target_2][feature].agg(['mean', 'median', 'std', 'min', 'max']).to_dict()
        
            # Add Prefixes to Distinguish the Groups
            target_one_renamed = {f'{target_1} {k}': v for k, v in target_one_stats.items()}
            target_two_renamed = {f'{target_2} {k}': v for k, v in target_two_stats.items()}
        
            # Combine the Stats
            summary_dict[group] = {**target_one_renamed, **target_two_renamed}
        

        # Convert to Dataframe
        df_summary_stats = pd.DataFrame(summary_dict).T

        # Round Stats
        df_summary_stats = df_summary_stats.round(2)

        # Convert Index to Column
        df_summary_stats = df_summary_stats.reset_index()

        # Renaming Index Column
        df_summary_stats = df_summary_stats.rename(columns={'index': f'{column_title}'})

        # Create Group Dataframe
        df_group = df_summary_stats[[f'{column_title}']]

        # Create Target Dataframes
        df_target_1 = df_summary_stats.iloc[:, 0:6]
        df_target_2 = df_summary_stats.iloc[:, 6:]
        df_target_2 =  pd.concat([df_group, df_target_2], axis=1)

        # Return Target Dataframes
        return df_target_1, df_target_2




    # Define Method for Group Comparisons
    def group_comparisons(self, data, column, feature, target):

        '''
        Method compares means for each subgroup and each target value.
        
        Parameters:
        -----------
        self: The VisualAnalysis object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The column to measure
        target: Column of the target of comparison

        Returns:
        --------
        df_comparisons: dataframe
        '''

        # Create List Variables
        groups = []
        comparisons = []

        # Create Data Lists
        group_list = data[column].value_counts().head(2).keys()
        target_list = data[target].value_counts().head(2).keys()

        # Set Target Variables
        target_1 = target_list[0]
        target_2 = target_list[1]

        # Create Title for Labels
        feature_title = self.titlize_column_name(feature)


        # Iterate Through Groups in Group List
        for group in group_list:

            # Filter Data by Group
            filtered_data = data[data.apply(lambda row: row[column] == group, axis=1)]
            
            # Calculate Target Means
            t1_mean = filtered_data[filtered_data[target] == target_1][feature].mean()
            t2_mean = filtered_data[filtered_data[target] == target_2][feature].mean()

            # Calculate Percent Difference Between Means
            percent_diff = abs((t2_mean - t1_mean) / t1_mean) * 100
        
            # et Comparison Value
            if t1_mean > t2_mean:
                comparison = "higher"
            else:
                comparison = "lower"
            
            # Append Lists
            groups.append(group)
            comparisons.append(f'{target_1} has {comparison} average {feature_title} values (by {percent_diff:.1f}%) than {target_2}')
            
        
        # Create Dataframe
        df_comparisons = pd.DataFrame({
            'Target Group': groups,
            'Key Observations from Comparisons': comparisons,
        })

        # Return Comparisons
        return df_comparisons




    # Define a Method for Statistical Summary
    def statistic_summary(self, column=None, feature=None, target=None, unit_test=None):

        '''
        Method creates summary statistics for each subgroup and each target value.
        
        Parameters:
        -----------
        self: The VisualAnalysis object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The column to measure
        target: Column of the target of comparison
        unit_test: Determines if unit tests are being performed

        Outputs:
        --------
        figure
        table
        '''

        # Set Data Variables
        data = self._data

        # Set Column Variable
        if column == None:
            column = self._column
        
        # Set Feature Variable
        if feature == None:
            feature = self._feature_1

        # Set Target Variable
        if target == None:
            target = self._target
        
        # Set Unit Test Variable
        if unit_test == None:
            unit_test = self._unit_test

        # Create Data Lists
        target_list = data[target].value_counts().head(2).keys()

        # Set Target Variables
        target_1 = target_list[0]
        target_2 = target_list[1]

        # Get Summary Stats
        summary_stats = self.get_summary_stats(data, column, feature, target)

        # Create Title for Labels
        column_title = self.titlize_column_name(column)
        target_title = self.titlize_column_name(target)
        feature_title = self.titlize_column_name(feature)

        # Plot Summary Stats Target 1
        if unit_test == False:
            stats_title = f'Statistical Summary of {feature_title} for {target_title} ({target_1}) by each {column_title}'
            self.plot_table_data(summary_stats[0], stats_title)

        # Plot Summary Stats Target 2
        if unit_test == False:
            stats_title = f'Statistical Summary of {feature_title} for {target_title} ({target_2}) by each {column_title}'
            self.plot_table_data(summary_stats[1], stats_title)

        # Perform Group Comparisons
        group_comparisons = self.group_comparisons(data, column, feature, target)
        target_comparisons = self.group_comparisons(data, target, feature, column)

        # Combine Dataframes
        df_comparisons = pd.concat([group_comparisons, target_comparisons])

        # Plot Group Comparisons
        if unit_test == False:
            comparisons_title = f'\nSummary of Comparison Analysis for {feature_title} by Target Group'
            col_widths = [0.2, 0.8]
            self.plot_table_data(df_comparisons, comparisons_title, col_widths, 'left')

        # Create List of Summary Statistics
        summary_stats_list = [summary_stats[0], summary_stats[1], df_comparisons]
        
        # Return Summary Statistics List
        if unit_test == True:
            return summary_stats_list










# End of Page
