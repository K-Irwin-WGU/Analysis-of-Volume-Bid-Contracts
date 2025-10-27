'''distribution_analysis'''


# IMPORT PACKAGES

import numpy as np 
import pandas as pd 

from prettytable import PrettyTable 

import math
from scipy import stats 


import seaborn as sns 
import matplotlib.pyplot as plt 

# Suppress Warnings
import warnings
warnings.filterwarnings('ignore')




# CREATE DISTRIBUTION ANALYSIS CLASS

class DistributionAnalysis():

    '''
    Class performs distribution analysis on data
    and returns a list of test results.

    Returns:
    --------
    distribution_results: list

    Outputs:
    --------
    figure
    histogram plot
    Q-Q plot
    box plot
    bar plot
    tables
    '''

    # Define init Method
    def __init__(self, data=None, column=None, 
                 feature=None, target=None, group='all', unit_test=False):

        '''
        Method initializes class variables.
        
        Parameters:
        -----------
        self: The DistributionAnalysis object
        self._data: Dataset to analyze
        self._column: Column for subgroups
        self._feature: The columns to measure
        self._target: Column of the target of comparison
        self._group: The subgroup for the analysis
        unit_test: Determines if unit tests are being performed
        '''

        # Initialize Class Variables
        self._data = data
        self._column = column
        self._feature = feature
        self._target = target
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
    

    # Getter for Target Variable
    @property
    def target(self):  
        return self._target
    

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
    



    # Define Setter for Target Variable
    @target.setter
    def target(self, target_column):
        if isinstance(target_column, str) and len(target_column) > 0:
            self._target = target_column
        else:
            raise ValueError('Target column must be a non-empty string.')
    



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
        self: The DistributionAnalysis object
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




    ############################## Plot Methods ################################



    # Define Method for Histogram Plot
    def histogram_plot(self, data, feature, ax):

        '''
        Method creates a histogram plot comparing actual values
        against normal values.
        
        Parameters:
        -----------
        self: The DistributionAnalysis object
        data: Dataset to analyze
        feature: The column to measure
        ax: The subplot figure

        Outputs:
        --------
        histogram
        '''

        # Create Title for Labels
        feature_title = self.titlize_column_name(feature)

        # Create Histogram
        sns.histplot(data, kde=True, ax=ax)

        # Calculate Normal Distribution
        mu, std = stats.norm.fit(data)
        x = np.linspace(min(data), max(data), 100)
        p = stats.norm.pdf(x, mu, std)

        # Add Normal Distribution Overlay
        ax.plot(x, p * len(data) * (max(data) - min(data)) / 100,
                'r-', linewidth=2, label=f'Normal: μ={mu:.2f}, σ={std:.2f}')

        # Add Mean and Standard Deviation Lines
        ax.axvline(mu, color='r', linestyle='--', alpha=0.8, label='Mean')
        ax.axvline(mu + std, color='g', linestyle='-.', alpha=0.8, label='μ + 1σ')
        ax.axvline(mu - std, color='g', linestyle='-.', alpha=0.8, label='μ - 1σ')

        # Add Border
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.25)
            ax.spines[axis].set_color("#010101")

        # Add Grid
        ax.grid(True, color = '#010101', linestyle = '--', linewidth = 0.5)

        # Add Title and Labels
        ax.set_title(f'Histogram of {feature_title} with Normal Curve')
        ax.set_xlabel(feature_title)
        ax.set_ylabel('Frequency')
        
        # Add Legend
        legend = ax.legend(shadow=True)
        legend.get_frame().set_edgecolor("#010101")




    # Define Method for Q-Q Plot
    def qq_polt(self, data, feature, ax):

        '''
        Method creates a Quantile-Quantile plot comparing actual values
        against theoretical values.
        
        Parameters:
        -----------
        self: The DistributionAnalysis object
        data: Dataset to analyze
        feature: The column to measure
        ax: The subplot figure

        Outputs:
        --------
        q-q plot
        '''

        # Create Title for Labels
        feature_title = self.titlize_column_name(feature)

        # Create Q-Q Plot
        stats.probplot(data, dist="norm", plot=ax)

        # Add Border
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.25)
            ax.spines[axis].set_color("#010101")

        # Add Grid
        ax.grid(True, color = '#010101', linestyle = '--', linewidth = 0.5)

        # Add Plot Title
        ax.set_title(f'Q-Q Plot for {feature_title}')
    



    # Define Method for Box Plot
    def box_plot(self, data, group, feature, ax):

        '''
        Method creates a box plot for the selected feature.
        
        Parameters:
        -----------
        self: The DistributionAnalysis object
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
        ax.set_title(f'Box Plot of {feature_title}')
        ax.set_xlabel(group)
        ax.set_ylabel(feature_title)




    # Define Method for Skewness and Kurtosis Plot
    def sk_plot(self, data, group, feature, ax):

        '''
        Method creates a box plot for visualizing
        skewness and kurtosis.
        
        Parameters:
        -----------
        self: The DistributionAnalysis object
        data: Dataset to analyze
        group: The subgroup for the analysis
        feature: The columns to measure
        ax: The subplot figure

        Outputs:
        --------
        box plot
        '''

        # Calculate Skewness and Kurtosis
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)

        # Create Labels and Metrics
        metrics = ['Skewness', 'Kurtosis']
        data_metrics = [skewness, kurtosis]

        # Create Bar Chart
        sns.barplot(x=metrics, hue=metrics, y=data_metrics, palette='viridis', ax=ax)
        ax.axhline(y=0, color='r', linestyle='--')

        # Add Values on Bars
        for i, v in enumerate(data_metrics):
            ax.text(i, v + 0.1 if v >= 0 else v - 0.2, f'{v:.3f}',
                    ha='center', va='center', fontsize=10)

        # Add Border
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.25)
            ax.spines[axis].set_color("#010101")

        # Add Grid
        ax.grid(True, axis = 'y', color = '#010101', linestyle = '--', linewidth = 0.5)
        
        # Create Title
        feature_title = self.titlize_column_name(feature)

        # Add Title and Labels
        ax.set_title(f'Skewness and Kurtosis for {feature_title}')
        ax.set_xlabel(group)
        ax.set_ylabel('Amount')




    ############################## Table Methods ###############################



    # Define Method for Plot Table
    def table_plot(self, dataframe, title, ax, col_widths=None, cell_location='right'):

        '''
        Method creates a table plot for the selected data.
        
        Parameters:
        -----------
        self: The DistributionAnalysis object
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
        self: The DistributionAnalysis object
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
    


    # Define Method for Shapiro-Wilk Test
    def shapiro_wilk_test(self, data, feature, target):

        '''
        Method performs the Shapiro-Wilk Test on subdivided data
        to test for normality.
        
        Parameters:
        -----------
        self: The DistributionAnalysis object
        data: Dataset to analyze
        feature: The columns to measure
        target: The column to subdivide data into groups

        Returns:
        --------
        df_test_results: dataframe
        test_averages: list
        '''


        # Select Key Target Values for Analysis
        group_list = data[target].value_counts().head(9).keys()
        group_list = sorted(group_list)

        # Create Lists for Test Results
        groups = []
        statistics = []
        p_values = []
        interpretations = []
        

        # Iterate Through Groups in Group List
        for group in group_list:

            # Filter Data to Reduce Sample Size
            filtered_data = data[data.apply(lambda row: row[target] == group, axis=1)]
            values = filtered_data[feature]
            
            # Perform Shapiro-Wilk Test
            test_stats, test_p_value = stats.shapiro(values)

            
            # Determine Normality
            if test_p_value < 0.05:
                t_interpretation = "Data does not look normally distributed"
            else:
                t_interpretation = "Data looks normally distributed"
            
            # Append Results to Lists
            groups.append(group[:2])
            statistics.append(test_stats)
            p_values.append(test_p_value)
            interpretations.append(t_interpretation)


        # Create Test Results DataFrame
        df_test_results = pd.DataFrame({
            'Target Group': groups,
            'Statistic': statistics,
            'P_Value': p_values,
            'Test Interpretation': interpretations
        })

        # Calculate Tests Average
        avg_statistic = (sum(statistics)/ len(statistics))
        avg_p_value = (sum(p_values) / len(p_values))

        # Determine Average Normality
        if avg_p_value < 0.05:
            avg_interpretation = "Reject H₀: Data does not look normally distributed"
        else:
            avg_interpretation = "Fail to reject H₀: Data looks normally distributed"
        
        # Create List of Average Test Results
        test_averages = [avg_statistic, avg_p_value, avg_interpretation]

        # Return Test Results
        return df_test_results, test_averages




    # Define Method for Calculating P-Value
    def calculate_pvalue(self, statistic):

        '''
        Method converts the Anderson-Darling statistic to a p-value.
        
        Parameters:
        -----------
        self: The DistributionAnalysis object
        statistic: The Anderson-Darling statistic to convert

        Returns:
        --------
        p_value: float
        '''

        # Calculate Adjusted Statistic
        adj_stat = statistic*(1 + (.75/50) + 2.25/(50**2))

        # Calculate P-Value
        if adj_stat >= .6:
            p_value = math.exp(1.2937 - 5.709*adj_stat - .0186*(adj_stat**2))
        elif adj_stat >=.34:
            p_value = math.exp(.9177 - 4.279*adj_stat - 1.38*(adj_stat**2))
        elif adj_stat >.2:
            p_value = 1 - math.exp(-8.318 + 42.796*adj_stat - 59.938*(adj_stat**2))
        else:
            p_value = 1 - math.exp(-13.436 + 101.14*adj_stat - 223.73*(adj_stat**2))

        # Return Results
        return p_value



    # Define Method for Anderson-Darling Test
    def anderson_darling_test(self, data):

        '''
        Method performs the Anderson-Darling Test on data
        to determine normality.
        
        Parameters:
        -----------
        self: The DistributionAnalysis object
        data: Dataset to analyze

        Returns:
        --------
        df_test_results: dataframe
        ad_statistic: float
        '''

        # Perform Anderson-Darling Test
        result = stats.anderson(data, dist='norm')

        # Set Statistic Variable
        ad_statistic = result.statistic

        # Set P-Value Variable
        ad_pvalue = self.calculate_pvalue(ad_statistic)
        
        # Create Lists for Test Results
        s_levels = []
        c_values = []
        interpretations = []
        

        # Itterate Through Critical Values
        for i in range(len(result.critical_values)):

            # Get Significance Level
            s_level = result.significance_level[i]

            # Get Critical Value
            c_value = result.critical_values[i]

            # Determine Normality
            if result.statistic > result.critical_values[i]:
                t_interpretation = 'Reject H₀: Data does NOT look normally distributed'
            else:
                t_interpretation = 'Fail to reject H₀: Data looks normally distributed'

            # Append Results to Lists
            s_levels.append(f'{s_level}%')
            c_values.append(c_value)
            interpretations.append(t_interpretation)
        

        # Create Test Results DataFrame
        df_test_results = pd.DataFrame({
            'Significance Level': s_levels,
            'Critical Value': c_values,
            'Test Interpretation': interpretations
        })

        # Return Test Results
        return df_test_results, ad_statistic, ad_pvalue




    # Define Method for Skewness
    def skewness_test(sef, data):

        '''
        Method performs skewness Test on data
        to determine the symmetry.
        
        Parameters:
        -----------
        self: The DistributionAnalysis object
        data: Dataset to analyze

        Returns:
        --------
        skewness: float
        t_interpretation: string
        '''

        # Calculate Skewness
        skewness = stats.skew(data)

        # Determine test Results
        if skewness < -0.5:
            t_interpretation = 'The distribution is negatively skewed (left-tailed)'
        elif abs(skewness) < 0.5:
            t_interpretation = 'The distribution is approximately symmetric'
        else:  # skewness > 0.5
            t_interpretation = 'The distribution is positively skewed (right-tailed)'

        # Return Test Results
        return skewness, t_interpretation
    



    # Define Method for Kurtosis Test
    def kurtosis_test(self, data):

        '''
        Method performs the Kurtosis Test on the data
        to measure the impact of the tails.
        
        Parameters:
        -----------
        self: The DistributionAnalysis object
        data: Dataset to analyze

        Returns:
        --------
        kurtosis: float
        t_interpretation: string
        '''

        # Calculate Kurtosis
        kurtosis = stats.kurtosis(data, fisher=True)

        # Determine Results
        if kurtosis < -0.5:
            t_interpretation = 'The data distribution has lighter tails than normal (platykurtic)'
        elif abs(kurtosis) < 0.5:
            t_interpretation = 'The data distribution has tails similar to normal distribution (mesokurtic)'
        else:  # kurtosis > 0.5
            t_interpretation = 'The data distribution has heavier tails than normal (leptokurtic)'

        # Return Test Results
        return kurtosis, t_interpretation




    ############################## Print Methods ###############################



    # Define Methon for Printing Shapiro-Wilk Test Results
    def print_shapiro_results(self, shapiro_results, target, section_header=None):

        '''
        Method prints the results of the Shapiro-Wilk Test.
        
        Parameters:
        -----------
        self: The DistributionAnalysis object
        shapiro_results: A dataframe and a list of test results
        target: The column of the subdivided data groups

        Returns:
        --------
        none

        Outputs:
        --------
        print statements
        results table
        average table
        '''

        # Create Title for Labels
        target_title = self.titlize_column_name(target)

        # Create Table From Results Dataframe
        results_table = self.create_table(shapiro_results[0], 1)

        # Create Dataframe for Shapiro Results
        df_shapiro_results = shapiro_results[0]

        # Format Columns to Six Decimal Places
        df_shapiro_results['Statistic'] = df_shapiro_results['Statistic'].apply(lambda x: f'{x:.6f}')
        df_shapiro_results['P_Value'] = df_shapiro_results['P_Value'].apply(lambda x: f'{x:.6f}')


        avg_statistic, avg_p_value, avg_interpretation = shapiro_results[1]

        # Create Dataframe for Average Results
        df_avg_results = pd.DataFrame({
            'Test Result': ['Average Statistic', 'Average P-Value'],
            'Value': [avg_statistic, avg_p_value]
        })

        # Format Column to Six Decimal Places
        df_avg_results['Value'] = df_avg_results['Value'].apply(lambda x: f"{x:.6f}")

        # Create Table From Average Dataframe
        avg_table = self.create_table(df_avg_results, 1)

        # Create Dataframe for Test Interpretation
        df_interpretation = pd.DataFrame({
            'Interpretation': [f'{avg_interpretation}']
        })

        # Print Statistical Test's Taregt for Subdivision
        results_title = f'Shapiro-Wilk Test Results by {target_title}'
        results_col_widths = [0.2, 0.2, 0.2, 0.4]
        self.plot_table_data(df_shapiro_results, results_title, results_col_widths, 'left')

        # Print Results Average
        avg_title = 'Shapiro-Wilk Test Average Results'
        avg_col_widths = [0.6, 0.4]
        self.plot_table_data(df_avg_results, avg_title, avg_col_widths, 'center')

        # Print Final Interpretation
        final_title = 'Overall Assessment of Shapiro-Wilk Test'
        final_col_widths = [1.0]
        self.plot_table_data(df_interpretation, final_title, final_col_widths, 'center')
    



    # Define Methon for Printing Anderson-Darling Test Results
    def print_anderson_results(self, anderson_results, section_header=None):

        '''
        Method prints the results of the Anderson-Darling Test.
        
        Parameters:
        -----------
        self: The DistributionAnalysis object
        anderson_results: A dataframe and a float of test results

        Returns:
        --------
        none

        Outputs:
        --------
        print statements
        statistics table
        results table
        '''

        # Set Anderson Results Variables
        df_test_results = anderson_results[0]
        ad_statistic = anderson_results[1]
        ad_pvalue = anderson_results[2]

        # Format Column to Six Decimal Places
        df_test_results['Critical Value'] = df_test_results['Critical Value'].apply(lambda x: f'{x:.6f}')

        # Create Table From Results Dataframe
        results_table = self.create_table(df_test_results, 1)

        # Create Dataframe for Test Statistic
        df_stats = pd.DataFrame({
            'Anderson Result': ['Statistic', 'P-Value'],
            'Value': [f'{ad_statistic:.4f}', f'{ad_pvalue:.6f}']
        })

        # Create Table From Statistic Dataframe
        stats_table = self.create_table(df_stats, 1)

        # Print Test Statistic
        stats_title = 'Anderson-Darling Test Result'
        stats_col_widths = [0.6, 0.4]
        self.plot_table_data(df_stats, stats_title, stats_col_widths, 'center')

        # Print Results Table
        results_title = 'Critical values at significance levels'
        results_col_widths = [0.2, 0.2, 0.6]
        self.plot_table_data(df_test_results, results_title, results_col_widths, 'left')
    



    # Define Methon for Printing Skewness and Kurtosis
    def print_skewness_kurtosis(self, skewness_results, kurtosis_results, section_header=None):

        '''
        Method prints the results of the skewness and kurtosis tests.
        
        Parameters:
        -----------
        self: The DistributionAnalysis object
        skewness_results: The results of the skewness test
        kurtosis_results: The results of the kurtosis test

        Returns:
        --------
        none

        Outputs:
        --------
        print statements
        results table
        '''

        # Set Skewness Variables
        skewness = skewness_results[0]
        s_interpretation = skewness_results[1]

        # Set Kurtosis Variables
        kurtosis = kurtosis_results[0]
        k_interpretation = kurtosis_results[1]

        # Create Dataframe for Test Results
        df_test_results = pd.DataFrame({
            'Test': ['Skewness', 'Kurtosis'],
            'Result': [skewness, kurtosis],
            'Interpretation': [s_interpretation, k_interpretation]
        })

        # Format Column to Six Decimal Places
        df_test_results['Result'] = df_test_results['Result'].apply(lambda x: f'{x:.6f}')

        # Create Table From Results Dataframe
        results_table = self.create_table(df_test_results, 1)

        # Print Results Table
        results_title = 'Overall Assessment of Skewness and Kurtosis'
        results_col_widths = [0.2, 0.2, 0.6]
        self.plot_table_data(df_test_results, results_title, results_col_widths, 'left')
        



    ############################## Run Methods #################################



    # Define Method to Plot Distribution of Data
    def plot_data_distribution(self, data, group, feature):

        '''
        Method creates a series of plots for visually
        analyzing the distribution of the data.
        
        Parameters:
        -----------
        self: The DistributionAnalysis object
        data: Dataset to analyze
        group: The subgroup for the analysis
        feature: The columns to measure

        Returns:
        --------
        none

        Outputs:
        --------
        figure
        histogram
        q-q plot
        box plot
        sk plot
        '''

        # Create Title for Labels
        feature_title = self.titlize_column_name(feature)

        # Set Figure Style
        sns.set_theme(style='whitegrid')
        plt.rcParams['axes.facecolor'] = "#F4F3F3FF"
        plt.rcParams['font.size'] = 12

        # Create Figure and Subplots
        _, axes = plt.subplots(2, 2, figsize=(14, 10))
        plt.subplots_adjust(hspace=0.4, wspace=0.4)

        # Create Plot 1: Histogram with Normal Curve
        ax1 = axes[0, 0]
        self.histogram_plot(data, feature, ax1)

        # Create Plot 2: Q-Q Plot
        ax2 = axes[0, 1]
        self.qq_polt(data, feature, ax2)

        # Create Plot 3: Box Plot
        ax3 = axes[1, 0]
        self.box_plot(data, group, feature, ax3)

        # Create Plot 4: Skewness and Kurtosis Plot
        ax4 = axes[1, 1]
        self.sk_plot(data, group, feature, ax4)

        # Display Plots
        plt.tight_layout()
        plt.suptitle(f'{group}  Distribution Analysis for {feature_title}', fontsize=16, y=1.02)
        plt.show()

    


    # Define Method to Print Distribution of Data
    def print_data_distribution(self, test_results, group, feature, target):

        '''
        Method creates a series of plots for visually
        analyzing the distribution of the data.
        
        Parameters:
        -----------
        self: The DistributionAnalysis object
        test_results: The results of the normality tests
        group: The subgroup of the analysis
        feature: The columns to measure
        target: The column of the subdivided data

        Returns:
        --------
        none

        Outputs:
        --------
        print statements
        shapiro results table
        shapiro average table
        anderson statistics table
        anderson results table
        s&k results table
        '''

        # Set Test Results Variables
        shapiro_results = test_results[0]
        anderson_results = test_results[1]
        skewness_results = test_results[2]
        kurtosis_results = test_results[3]

        # Create Title for Labels
        feature_title = self.titlize_column_name(feature)

        # Create Section Header for Statistical Tests
        section_header = f'Statistical Results for {feature_title} in the {group} Group\n'

        # Print Shapiro-Wilk Results
        self.print_shapiro_results(shapiro_results, target, section_header)

        # Print Anderson-Darling Results
        self.print_anderson_results(anderson_results)

        # Print Skewness Results
        self.print_skewness_kurtosis(skewness_results, kurtosis_results)




    # Define Method for Distribution Analysis
    def distribution_analysis(self, data, group, feature, target, unit_test):
        
        '''
        Method utilizes several techniques to visualize and analyze 
        the data to determine its normality, skewness, and kurtosis.
        
        Parameters:
        -----------
        self: The DistributionAnalysis object
        data: Dataset to analyze
        group: The subgroup for the analysis
        feature: The columns to measure
        target: The column to subdivide data into groups
        unit_test: Determines if unit tests are being performed

        Returns:
        --------
        distribution_results: list

        Outputs:
        --------
        graphs
        tables
        print statements
        '''

        # Set Data Values
        values = data[feature]

        # Create Plots for Visual Analysis of Distribution
        if unit_test == False:
            self.plot_data_distribution(values, group, feature)

        # Perform Shapiro-Wilk Test
        shapiro_results = self.shapiro_wilk_test(data, feature, target)

        # Perform Anderson-Darling Test
        anderson_results = self.anderson_darling_test(values)

        # Calculate Skewness
        skewness_results = self.skewness_test(values)

        # Calculate Kurtosis
        kurtosis_results = self.kurtosis_test(values)

        # Create Distribution Results List
        distribution_results = [
            shapiro_results, anderson_results, 
            skewness_results, kurtosis_results
        ]

        # Print Distribution Results
        if unit_test == False:
            self.print_data_distribution(distribution_results, group, feature, target)

        # Return Test Results
        return distribution_results




    ############################## Main Method #################################



    # Define Methon to Apply Distribution Analysis
    def apply_distribution_analysis(self, column=None, feature=None, target=None, group=None, unit_test=None):

        '''
        Method performs distribution analysis on all groups in the selected column
        to determine its normality, skewness, and kurtosis.
        
        Parameters:
        -----------
        self: The DistributionAnalysis object
        column: Column for subgroups
        feature: The columns to measure
        target: The column to subdivide data into groups
        group: The subgroup for the analysis
        unit_test: Determines if unit tests are being performed

        Returns:
        --------
        distribution_results: list

        Outputs:
        --------
        graphs
        tables
        print statements
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
        
        # Set Group Variable
        if group == None:
            group = self._group
        
        # Set Unit Test Variable
        if unit_test == None:
            unit_test = self._unit_test
        
        # Select Key Column Values for Analysis
        group_list = data[column].value_counts().head(8).keys()

         # Set Group Variable
        group_1 = group_list[0]
        group_2 = group_list[1]

        if group == 'first':

            # Print Group
            #print(f'\nAnalysis of Normality in {group_1} data\n')
            print()

            # Filter Data
            filtered_data = data[data.apply(lambda row: row[column] == group_1, axis=1)]

            # Perform Normality Analysis
            distribution_results = self.distribution_analysis(filtered_data, group_1, feature, target, unit_test)
        
        elif group == "second":

            # Print Group
            #print(f'\nAnalysis of Normality in {group_2} data\n')
            print()

            # Filter Data
            filtered_data = data[data.apply(lambda row: row[column] == group_2, axis=1)]

            # Perform Normality Analysis
            distribution_results = self.distribution_analysis(filtered_data, group_2, feature, target, unit_test)

        else:
            # Create Test Results List
            distribution_results = []

            # Iterate Through Each Group in Group List
            for subgroup in group_list:

                # Print Group
                print(f'\nAnalysis of Normality in {subgroup}\n')

                # Filter Data
                filtered_data = data[data.apply(lambda row: row[column] == subgroup, axis=1)]

                # Perform Normality Analysis
                results = self.distribution_analysis(filtered_data, subgroup, feature, target, unit_test)

                # Appent to List
                distribution_results.append(results)

        # Return Distribution Results
        return distribution_results















# End of Page
