'''t_test_analysis'''


# IMPORT PACKAGES

import numpy as np
import pandas as pd

from prettytable import PrettyTable

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.power import TTestIndPower

from transformation_analysis import TransformationAnalysis

# Suppress Warnings
import warnings
warnings.filterwarnings('ignore')




# CREATE INDEPENDENT T-TEST ANALYSIS CLASS

class TTestAnalysis():

    '''
    Class performs an independent samples t-test and returns the results.

    Returns:
    --------
    t_test_results: dictionary

    Outputs:
    --------
    box plot
    violin plot
    histogram
    bar plot
    tables
    '''

    # Define init Method
    def __init__(self, data=None, column=None, feature=None, transform=False, unit_test=False):

        '''
        Method initializes class variables.
        
        Parameters:
        -----------
        self: The TTestAnalysis object
        self._data: Dataset to analyze
        self._column: Column for subgroups
        self._feature: The columns to measure
        self._transform: Determines if the data is transformed
        unit_test: Determines if unit tests are being performed
        '''

        # Initialize Class Variables
        self._data = data
        self._column = column
        self._feature = feature
        self._transform = transform
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
    

    # Getter for Transform Variable
    @property
    def transform(self):  
        return self._transform
    

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
    



    # Define Setter for Feature Variable
    @transform.setter
    def transform(self, transform_value):
        if isinstance(transform_value, bool):
            self._transform = transform_value
        else:
            raise ValueError('Transform must be True or False.')
    



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
        self: The TTestAnalysis object
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
        self: The TTestAnalysis object
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
        self: The TTestAnalysis object
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



    # Define Method for Box Plot
    def box_plot(self, data, column, feature, ax):

        '''
        Method creates a box plot comparing the two datasets.
        
        Parameters:
        -----------
        self: The TTestAnalysis object
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
        self: The TTestAnalysis object
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
        self: The TTestAnalysis object
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
    def bar_plot(self, data, column, feature, p_value, ax):

        '''
        Method creates a bar plot comparing the two datasets.
        
        Parameters:
        -----------
        self: The TTestAnalysis object
        data: The dataframe with the dataset stats
        column: Column for subgroups
        feature: The column to measure
        p_value: The p-value from the Independent T-Test
        ax: The subplot figure

        Returns:
        --------
        None

        Outputs:
        --------
        bar plot
        '''

        # Create Title for Labels
        column_title = self.titlize_column_name(column)
        feature_title = self.titlize_column_name(feature)

        # Create Color Palette
        palette = 'viridis'

        # Create Bar Plot
        sns.barplot(x='Group', hue='Group', y='Mean', data=data, palette=palette, ax=ax)

        # Add Error Bars
        ax.errorbar(x=range(len(data)), y=data['Mean'],
                    yerr=[(data['Mean'] - data['CI_Lower']).values,
                        (data['CI_Upper'] - data['Mean']).values],
                    fmt='none', c='black', capsize=5)


        # Add Significance Annotation
        if p_value < 0.05:
            max_y = max(data['CI_Upper']) * 1.1
            ax.plot([0, 0, 1, 1], [max_y, max_y + 0.05*max_y, max_y + 0.05*max_y, max_y], 'k-')
            ax.text(0.5, max_y + 0.05*max_y, f'p = {p_value:.4f}', ha='center')

        # Add Border
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.25)
            ax.spines[axis].set_color("#010101")

        # Add Grid
        ax.grid(True, axis = 'y', color = '#010101', linestyle = '--', linewidth = 0.5)

        # Add Title and Labels
        ax.set_title(f'Mean {feature_title} with 95% CI by {column_title}')
        ax.set_xlabel(column_title)
        ax.set_ylabel(f'Mean {feature_title}')




    # Define Method to Plot Data
    def plot_ttest_analysis(self, data, column, feature, dataframe, p_value, transform):

        '''
        Method creates a figure displaying a series of plots
        for comparing the two datasets used in testing.
        
        Parameters:
        -----------
        self: The TTestAnalysis object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The column to measure
        dataframe: The dataframe with the dataset stats
        p_value: The p-value from the Independent T-Test
        transform: Determines if the data is transformed

        Returns:
        --------
        None

        Outputs:
        --------
        figure
        box plot
        violin plot
        bar plot
        '''

        # Create Title for Labels
        column_title = self.titlize_column_name(column)
        feature_title = self.titlize_column_name(feature)

        # Set Data_Form
        if transform == False:
            data_form = 'Original'
        else:
            data_form = 'Transformed'

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

        # Create Plot 4: Bar Plot
        ax4 = axes[1, 1]
        self.bar_plot(dataframe, column, feature, p_value, ax4)

        # Display Plots
        plt.tight_layout()
        plt.suptitle(f'Independent Samples T-Test: {feature_title} by {column_title} - {data_form} Data Format', fontsize=16, y=1.02)
        plt.show()




    ############################## Table Methods ###############################



    # Define Method for Plot Table
    def table_plot(self, dataframe, title, ax, col_widths=None, cell_location='right'):

        '''
        Method creates a table plot for the selected data.
        
        Parameters:
        -----------
        self: The TTestAnalysis object
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
        self: The TTestAnalysis object
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
    



    # Define Method for Equal Variance Analysis
    def equal_variance_analysis(self, data, column, feature):

        '''
        Method performs Levene's test to assess the equality of variances
        across the two datasets.
        
        Parameters:
        -----------
        self: The TTestAnalysis object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The column to measure

        Returns:
        --------
        p_levene: float
        equal_variance: boolean variable
        '''

        # Select Key Column Values for Analysis
        group_list = data[column].unique()

        # Set Group Variable
        group_1 = group_list[0]
        group_2 = group_list[1]

        # Split the Data into Two Groups
        dataset_1 = data[data[column] == group_1][feature].dropna()
        dataset_2 = data[data[column] == group_2][feature].dropna()

        # Perform Levene's Test for Equal Variance
        _, p_levene = stats.levene(dataset_1, dataset_2)


        # Determine Equal Variance Based on Alpha
        if p_levene < 0.05:
            equal_variance = False
        else:
            equal_variance = True
        
        # Return Results
        return p_levene, equal_variance
    



    # Define Method to Calculate Statistical Data
    def stats_data(self, data, column, feature):

        '''
        Method creates a dataframe with the statistics
        of the two datasets.
        
        Parameters:
        -----------
        self: The TTestAnalysis object
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
        Mean: The means to the two datasets
        STD: The standard deviations of the two datasets
        SEM: The standard error of the means of the two datasets
        CI_Lower: The lower confidence intervals
        CI_Upper: The upper confidence intervals
        '''

        # Select Key Column Values for Labels
        group_list = data[column].unique()

        # Split the Data into Two Groups
        dataset_1, dataset_2 = self.split_data(data, column, feature)

        # Calculate Descriptive Statistics
        mean_1, std_1 = dataset_1.mean(), dataset_1.std()
        mean_2, std_2 = dataset_2.mean(), dataset_2.std()
        n1, n2 = len(dataset_1), len(dataset_2)

        # Determine Equal Variance
        _, equal_variance = self.equal_variance_analysis(data, column, feature)

        # Calculate Degrees of Freedom
        if equal_variance == True:
            dof = n1 + n2 - 2
        else:
            dof = ((std_1**2/n1 + std_2**2/n2)**2) / \
                ((std_1**4/(n1**2*(n1-1)) + std_2**4/(n2**2*(n2-1))))

        # Calculate 95% Lower-Tail Probability
        t_value = stats.t.ppf(0.975, dof, loc=0, scale=1)

        # Create Dataframe for Stats Data
        df_stats_data = pd.DataFrame({
            'Group': [group_list[0], group_list[1]],
            'Size': [n1, n2],
            'Mean': [mean_1, mean_2],
            'STD': [std_1, std_2],
            'SEM': [std_1/np.sqrt(n1), std_2/np.sqrt(n2)],
            'CI_Lower': [mean_1 - t_value * std_1/np.sqrt(n1), mean_2 - t_value * std_2/np.sqrt(n2)],
            'CI_Upper': [mean_1 + t_value * std_1/np.sqrt(n1), mean_2 + t_value * std_2/np.sqrt(n2)]
        })

        # Return Dataframe
        return df_stats_data
    



    # Define Method for Effect Size
    def effect_size(self, data, column, feature):

        '''
        Method calculates the size effect of the difference between
        the two datasets with Cohen's d.
        
        Parameters:
        -----------
        self: The TTestAnalysis object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The column to measure

        Returns:
        --------
        d: float
        d_interpretation: string
        '''

        # Determine Equal Variance
        _, equal_variance = self.equal_variance_analysis(data, column, feature)

        # Calculate Descriptive Statistics
        df_stats = self.stats_data(data, column, feature)

        # Create Lists from DataFrame Column Values
        n1, n2 = df_stats['Size'].tolist()
        mean_1, mean_2 = df_stats['Mean'].tolist()
        std_1, std_2 = df_stats['STD'].tolist()

         # Calculate the Difference Between Group Means
        if equal_variance == True:
            # Used Pooled Standard Deviation
            std_pooled = np.sqrt(((n1 - 1) * std_1**2 + (n2 - 1) * std_2**2) / (n1 + n2 - 2))
            d = (mean_2 - mean_1) / std_pooled
        else:
            # Use Average Standard Deviation
            d = (mean_2 - mean_1) / np.sqrt((std_1**2 + std_2**2) / 2)

        # Interpret Effect Size
        if abs(d) < 0.2:
            d_interpretation = "negligible"
        elif abs(d) < 0.5:
            d_interpretation = "small"
        elif abs(d) < 0.8:
            d_interpretation = "medium"
        else:
            d_interpretation = "large"

        # Return Results
        return d, d_interpretation
    



    # Define Method for Mean Difference
    def mean_difference(self, data, column, feature):

        '''
        Method calculates the mean difference, lower confidence interval,
        and the upper confidence interval.
        
        Parameters:
        -----------
        self: The TTestAnalysis object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The column to measure

        Returns:
        --------
        mean_diff: float
        ci_lower: float
        ci_upper: float
        '''

        # Determine Equal Variance
        _, equal_variance = self.equal_variance_analysis(data, column, feature)

        # Calculate Descriptive Statistics
        df_stats = self.stats_data(data, column, feature)

        # Create Lists from DataFrame Column Values
        n1, n2 = df_stats['Size'].tolist()
        mean_1, mean_2 = df_stats['Mean'].tolist()
        std_1, std_2 = df_stats['STD'].tolist()

        # Calculate Degrees of Freedom
        if equal_variance == True:
            dof = n1 + n2 - 2
        else:
            dof = ((std_1**2/n1 + std_2**2/n2)**2) / \
                ((std_1**4/(n1**2*(n1-1)) + std_2**4/(n2**2*(n2-1))))


        # Calculate Mean Difference
        mean_diff = mean_2 - mean_1

        # Calculate Standard Error Difference
        if equal_variance == True:
            std_pooled = np.sqrt(((n1 - 1) * std_1**2 + (n2 - 1) * std_2**2) / (n1 + n2 - 2))
            se_diff = std_pooled * np.sqrt(1/n1 + 1/n2)
        else:
            se_diff = np.sqrt(std_1**2/n1 + std_2**2/n2)


        # Calculate 95% Lower-Tail Probability
        t_value = stats.t.ppf(0.975, dof, loc=0, scale=1)

        # Calculate Confidence Interval
        ci_lower = mean_diff - t_value * se_diff
        ci_upper = mean_diff + t_value * se_diff

        # Return Results
        return mean_diff, ci_lower, ci_upper
    



    # Define Method to Calculate Statistical Power
    def calculate_power(self, data, column, feature):

        '''
        Method calculates the statistical power of the dataset.
        
        Parameters:
        -----------
        self: The TTestAnalysis object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The column to measure

        Returns:
        --------
        power: float
        '''

        # Calculate Descriptive Statistics
        df_stats = self.stats_data(data, column, feature)

        # Create Lists from DataFrame Column Values
        n1, n2 = df_stats['Size'].tolist()

        # Calculate Effect Size
        es = self.effect_size(data, column, feature)
        d = es[0]

        # Set Power Analysis Method
        power_analysis = TTestIndPower()
        
        # Perform Power anaysis
        power = power_analysis.power(effect_size=abs(d), nobs1=n1, alpha=0.05, ratio=n2/n1)
        
        # Return Results
        return power




    ############################## Print Methods ###############################



    # Define Methon for Printing Results
    def print_results(self, data, column, feature, t_stat, p_value, transform):

        '''
        Method prints the statistical results of the independent t-test.
        
        Parameters:
        -----------
        self: The TTestAnalysis object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The column to measure
        t_stat: The T-statistic from the independent t-test
        p-value: The p-value from the independent t-test
        transform: Determines if the data is transformed

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

        # Create Lists from DataFrame Column Values
        group_1, group_2 = df_stats['Group'].tolist()

        # Calculate Effect Size
        d, d_interpretation = self.effect_size(data, column, feature)

        # Calculate Mean Difference
        mean_diff, ci_lower, ci_upper = self.mean_difference(data, column, feature)

        # Perform Power Analysis
        power = self.calculate_power(data, column, feature)

        # Create Dataframe for Results Data
        df_results_data = pd.DataFrame({
            'Statistic': [
                'Mean Difference',
                'Lower CI',
                'Upper CI',
                'T_Statistic',
                'P_Value',
                "Cohen's d",
                'Interpretation',
                'Power'
            ],
            'Value': [
                f'{mean_diff:.2f}',
                f'{ci_lower:.2f}',
                f'{ci_upper:.2f}',
                f'{t_stat:.4f}',
                f'{p_value:.4f}',
                f'{d:.4f}',
                f'{d_interpretation} effect',
                f'{power:.4f}'
            ]
        })

        # Create Table From Stats Dataframe
        results_table = self.create_table(df_results_data, 1)

        # Determine Final T-Test Assessment
        if p_value < 0.05:
            t_interpretation = f'There is a statistically significant difference in {feature_title} between {group_1} and {group_2}'
        else:
            t_interpretation = f'There is no statistically significant difference in {feature_title} between {group_1} and {group_2}'
        
        # Create Dataframe for T-Test Interpretation
        df_interpretation = pd.DataFrame({
            'Interpretation': [f'{t_interpretation}']
        })

        # Set Data_Form
        if transform == False:
            data_form = 'Original'
        else:
            data_form = 'Transformed'

        
        # Print Stats Table
        stats_title = f'Comparing {feature_title} between {group_1} and {group_2} - {data_form} Data Format'
        stats_col_widths = [0.15, 0.1, 0.15, 0.15, 0.15, 0.15, 0.15]
        self.plot_table_data(df_stats.round(6), stats_title, stats_col_widths, 'left')

        
        # Print Results Table
        results_title = f'Independent Samples T-Test Results'
        results_col_widths = [0.5, 0.5]
        self.plot_table_data(df_results_data, results_title, results_col_widths, 'left')


        # Print Final T-Test Interpretation
        final_title = f'Overall Assessment of Independent T-Test Analysis - {data_form} Data Format'
        final_col_widths = [1.0]
        self.plot_table_data(df_interpretation, final_title, final_col_widths, 'center')




    ############################## Main Method #################################



    # Define Method for Independent T-Test Analysis
    def independent_ttest_analysis(self, column=None, feature=None, transform=None, unit_test=None):

        '''
        Method performs an independent samples t-test and returns the results.
        
        Parameters:
        -----------
        self: The TTestAnalysis object
        column: Column for subgroups
        feature: The column to measure
        transform: Determines if the data is transformed
        unit_test: Determines if unit tests are being performed

        Returns:
        --------
        t_test_results: dictionary

        Outputs:
        --------
        box plot
        violin plot
        histogram
        bar plot
        tables
        '''

        # Set Column Variable
        if column == None:
            column = self._column

        # Set Feature Variable
        if feature == None:
            feature = self._feature

        # Set Transform Variable
        if transform == None:
            transform = self._transform
        
        # Set Unit Test Variable
        if unit_test == None:
            unit_test = self._unit_test

        # Set Data Variable
        if transform == True:
            _, data = self.transformed_data(self._data, column, feature)
        else:
            data = self._data

        # Split the Data into Two Groups
        dataset_1, dataset_2 = self.split_data(data, column, feature)
        
        # Perform Equal Variance Analysis
        _, equal_variance = self.equal_variance_analysis(data, column, feature)

        # Perform the T-Test
        t_stat, p_value = stats.ttest_ind(dataset_1, dataset_2, equal_var=equal_variance)

        # Calculate Effect Size
        es = self.effect_size(data, column, feature)
        d = es[0]

        # Calculate Mean Difference
        mean_diff, ci_lower, ci_upper = self.mean_difference(data, column, feature)

        # Perform Power Analysis
        power = self.calculate_power(data, column, feature)

        # Plot Data
        if unit_test == False:
            df_stats = self.stats_data(data, column, feature)
            self.plot_ttest_analysis(data, column, feature, df_stats, p_value, transform)

        # Print Results
        if unit_test == False:
            self.print_results(data, column, feature, t_stat, p_value, transform)

        # Create Results Dictionary
        t_test_results = {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': d,
            'mean_difference': mean_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'power': power
        }

        # Return Results Dictionary
        return t_test_results















# End of Page
