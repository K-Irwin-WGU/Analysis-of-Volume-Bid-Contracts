'''transformation_analysis'''


# IMPORT PACKAGES

import numpy as np
import pandas as pd

from prettytable import PrettyTable

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import normaltest

from sklearn.preprocessing import PowerTransformer, QuantileTransformer

# Suppress Warnings
import warnings
warnings.filterwarnings('ignore')




# CREATE TRANSFORMATION ANALYSIS CLASS

class TransformationAnalysis():

    '''
    Class performs transformation analysis on data.

    Returns:
    --------
    transformation_results: list

    Outputs:
    --------
    figure
    histogram plots
    Q-Q plots
    tables
    '''

    # Define init Method
    def __init__(self, data=None, column=None, feature=None, group='all', unit_test=False):

        '''
        Method initializes class variables.
        
        Parameters:
        -----------
        self: The TransformationAnalysis object
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
        self: The TransformationAnalysis object
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
    def create_table(self, dataframe):

        '''
        Method creates a table from a dataframe for printing purposes.
        
        Parameters:
        -----------
        self: The TransformationAnalysis object
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
        table.padding_width = 6

        # Return Table
        return table




    ############################## Plot Methods ################################



    # Define Method for Histogram Plot
    def histogram_plot(self, data, group, feature, ax):

        '''
        Method creates a histogram plot comparing actual values
        against normal values.
        
        Parameters:
        -----------
        self: The TransformationAnalysis object
        data: Dataset to analyze
        group: The subgroup from the subgroup column
        feature: The column to measure
        ax: The subplot figure

        Outputs:
        --------
        graph
        '''

        # Create Title for Labels
        feature_title = self.titlize_column_name(feature)

        # Calculate Measures of Central Tendency
        mean_val = data.mean()
        median_val = np.median(data)
        mode_val = stats.mode(data)[0]

        # Calculate Skew and Kurtosis
        skew = stats.skew(data)
        kurt = stats.kurtosis(data)

        # Create Histogram
        sns.histplot(data, kde=True, ax=ax)

        # Add Mean and Standard Deviation Lines
        ax.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='g', linestyle='-.', label=f'Median: {median_val:.2f}')
        ax.axvline(mode_val, color='b', linestyle=':', label=f'Mode: {mode_val:.2f}')

        # Add Border
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.25)
            ax.spines[axis].set_color("#010101")

        # Add Grid
        ax.grid(True, color = '#010101', linestyle = '--', linewidth = 0.5)

        # Add Title and Labels
        ax.set_title(f'{group}: {feature_title}\nSkewness: {skew:.4f}, Kurtosis: {kurt:.4f}')
        ax.set_xlabel(feature_title)
        ax.set_ylabel('Frequency')
        
        # Add Legend
        legend = ax.legend(shadow=True)
        legend.get_frame().set_edgecolor("#010101")




    # Define Method for Q-Q Plot
    def qq_polt(self, data, group, feature, ax):

        '''
        Method creates a Quantile-Quantile plot comparing actual values
        against theoretical values.
        
        Parameters:
        -----------
        self: The TransformationAnalysis object
        data: Dataset to analyze
        group: The subgroup from the subgroup column
        feature: The column to measure
        ax: The subplot figure

        Outputs:
        --------
        graph
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
        ax.set_title(f'{group} Q-Q Plot for {feature_title}')




    # Define Method to Plot Transformation Data
    def plot_transformations(self, data, group, feature, xform):

        '''
        Method creates a figure displaying a box plot
        and a histogram for visual analysis of data.
        
        Parameters:
        -----------
        self: The OutlierAnalysis object
        data: Dataset to analyze
        group: The subgroup for the analysis
        feature: The column to measure
        xform: The Transformation Used on the Dataset

        Outputs:
        --------
        figure
        graph
        '''

        # Print Empty Line for Spacing
        print()

        # Set Figure Style
        sns.set_theme(style='whitegrid')
        plt.rcParams['axes.facecolor'] = "#F4F3F3FF"
        plt.rcParams['font.size'] = 12

        # Create Figure
        _, axes = plt.subplots(1, 2, figsize=(15, 4))

        # Create Histogram Plot of Data Value
        ax1 = axes[0]
        self.histogram_plot(data, xform, feature, ax1)

        # Create Q-Q Plot of Data Values
        ax2 = axes[1]
        self.qq_polt(data, xform, feature, ax2)

        # Create Title for Labels
        feature_title = self.titlize_column_name(feature)

         # Display Plots
        plt.tight_layout()
        plt.suptitle(f'{xform} Analysis for {feature_title} in {group}', fontsize=16, y=1.02)
        plt.show()




    ############################## Table Methods ###############################



    # Define Method for Plot Table
    def table_plot(self, dataframe, title, ax, col_widths=None, cell_location='right'):

        '''
        Method creates a table plot for the selected data.
        
        Parameters:
        -----------
        self: The TransformationAnalysis object
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
        self: The TransformationAnalysis object
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




    ############################## Transformation Methods ######################



    # Define Method for Log Transformation
    def log_transformation(self, data):

        '''
        Method performs the Log transformation on the dataset.
        
        Parameters:
        -----------
        self: The TransformationAnalysis object
        data: The dataset for the transformation

        Returns:
        --------
        data_xform: dataset
        skew_xform: float
        kurt_xform : float
        '''

        # Check Minimum Value
        min_val = data.min()

        # Set Number for Constant
        num = 1 if min_val <= 0 else 0

        # Perform Transformation
        data_xform = np.log(data + num)

        # Calculate Skew
        skew_xform = stats.skew(data_xform)

        # Calculate Kurtosis
        kurt_xform = stats.kurtosis(data_xform)

        # Return Results
        return data_xform, skew_xform, kurt_xform
    



    # Define Method for Square Root Transformation
    def sqrt_transformation(self, data):

        '''
        Method performs the Square Root transformation on the dataset.
        
        Parameters:
        -----------
        self: The TransformationAnalysis object
        data: The dataset for the transformation

        Returns:
        --------
        data_xform: dataset
        skew_xform: float
        kurt_xform : float
        '''

        # Check Minimum Value
        min_val = data.min()

        # Set Number for Constant
        num = 1 if min_val <= 0 else 0

        # Perform Transformation
        data_xform = np.sqrt(data + num)

        # Calculate Skew
        skew_xform = stats.skew(data_xform)

        # Calculate Kurtosis
        kurt_xform = stats.kurtosis(data_xform)

        # Return Results
        return data_xform, skew_xform, kurt_xform
    



    # Define Method for Reciprocal Transformation
    def reciprocal_transformation(self, data):

        '''
        Method performs the Reciprocal transformation on the dataset.
        
        Parameters:
        -----------
        self: The TransformationAnalysis object
        data: The dataset for the transformation

        Returns:
        --------
        data_xform: dataset
        skew_xform: float
        kurt_xform : float
        '''

        # Check Minimum Value
        min_val = data.min()

        # Set Number for Constant
        num = 1 if min_val <= 0 else 0

        # Perform Transformation
        data_xform = 1 / (data + num)

        # Calculate Skew
        skew_xform = stats.skew(data_xform)

        # Calculate Kurtosis
        kurt_xform = stats.kurtosis(data_xform)

        # Return Results
        return data_xform, skew_xform, kurt_xform
    



    # Define Method for Box-Cox Transformation
    def box_cox_transformation(self, data):

        '''
        Method performs the Box-Cox transformation on the dataset.
        
        Parameters:
        -----------
        self: The TransformationAnalysis object
        data: The dataset for the transformation

        Returns:
        --------
        data_xform: dataset
        skew_xform: float
        kurt_xform : float
        '''

        # Check Minimum Value
        min_val = data.min()

        # Ensure All Values Are Positive
        if min_val <= 0:
            data_for_xform = (data - min_val) + 0.01
        else:
            data_for_xform = data

        try:
            # Perform Transformation
            data_xform, lambda_xform = stats.boxcox(data_for_xform)

            # Calculate Skew
            skew_xform = stats.skew(data_xform)

            # Calculate Kurtosis
            kurt_xform = stats.kurtosis(data_xform)
        
        except:
            data_xform = None
            skew_xform = None
            kurt_xform = None

        # Return Results
        return data_xform, skew_xform, kurt_xform
    



    # Define Method for Yeo-Johnson Transformation
    def yeo_johnson_transformation(self, data):

        '''
        Method performs the Yeo-Johnson transformation on the dataset.
        
        Parameters:
        -----------
        self: The TransformationAnalysis object
        data: The dataset for the transformation

        Returns:
        --------
        data_xform: dataset
        skew_xform: float
        kurt_xform : float
        '''

        # Create PowerTransformer Object
        pt = PowerTransformer(method='yeo-johnson')

        # Perform Transformation
        data_xform = pt.fit_transform(data.values.reshape(-1, 1)).flatten()

        # Calculate Skew
        skew_xform = stats.skew(data_xform)

        # Calculate Kurtosis
        kurt_xform = stats.kurtosis(data_xform)

        # Return Results
        return data_xform, skew_xform, kurt_xform
    



    # Define Method for Quantile Transformation
    def quantile_transformation(self, data):

        '''
        Method performs the Quantile transformation on the dataset.
        
        Parameters:
        -----------
        self: The TransformationAnalysis object
        data: The dataset for the transformation

        Returns:
        --------
        data_xform: dataset
        skew_xform: float
        kurt_xform : float
        '''

        # Create Quantile Transformation Object
        qt = QuantileTransformer(output_distribution='normal', random_state=42)

        # Perform Transformation
        data_xform =qt.fit_transform(data.values.reshape(-1, 1)).flatten()

        # Calculate Skew
        skew_xform = stats.skew(data_xform)

        # Calculate Kurtosis
        kurt_xform = stats.kurtosis(data_xform)

        # Return Results
        return data_xform, skew_xform, kurt_xform
    



    ############################## Transformation Run Method ####################



    # Define Methed to Run Data Transformations
    def data_transformations(self, data, feature):

        '''
        Method performs data transformations on the dataset.
        The method returns the transformed dataset and a dataframe
        containing the transformation statistics.
        
        Parameters:
        -----------
        self: The TransformationAnalysis object
        data: The dataset for the transformations
        feature: The columns to measure

        Returns:
        --------
        df_summary: dataframe
        data_list: list
        '''


        # Original Data
        data_orig = data[feature].copy()
        skew_orig = stats.skew(data_orig)
        kurt_orig = stats.kurtosis(data_orig)

        
        # Perform Log Transformation (log(x))
        data_log, skew_log, kurt_log = self.log_transformation(data_orig)
        
        # Perform Square Root Transformation (sqrt(x))
        data_sqrt, skew_sqrt, kurt_sqrt = self.sqrt_transformation(data_orig)
        
        # Perform Reciprocal Transformation (1/x)
        data_recip, skew_recip, kurt_recip = self.reciprocal_transformation(data_orig)
        
        # Perform Box-Cox Transformation
        data_box_cox, skew_box_cox, kurt_box_cox = self.box_cox_transformation(data_orig)

        # Perform Yeo-Johnson Transformation
        data_yeo_johnson, skew_yeo_johnson, kurt_yeo_johnson = self.yeo_johnson_transformation(data_orig)
        

        # Create List of Transformations
        transformations = ['Original', 'Log', 'Square Root', 'Reciprocal', 'Box-Cox', 'Yeo-Johnson']

        # Create List of Skewness Values
        try:
            skewness_values = [skew_orig, skew_log, skew_sqrt, skew_recip, skew_box_cox, skew_yeo_johnson]
        except:
            skewness_values = [skew_orig, skew_log, skew_sqrt, skew_recip, None, skew_yeo_johnson]

        # Create List of Kurtosis Values
        try:
            kurtosis_values = [kurt_orig, kurt_log, kurt_sqrt, kurt_recip, kurt_box_cox, kurt_yeo_johnson]
        except:
            kurtosis_values = [kurt_orig, kurt_log, kurt_sqrt, kurt_recip, None, kurt_yeo_johnson]


        # Summary DataFrame
        df_summary = pd.DataFrame({
            'Transformation': transformations,
            'Skewness': skewness_values,
            'Kurtosis': kurtosis_values
        })

        # Create List of Transformation Data
        data_list = [data_orig, data_log, data_sqrt, data_recip, data_box_cox, data_yeo_johnson]

        # Return Results
        return df_summary, data_list




    ############################## Normality Methods ############################



    # Define Method for Normality Analysis
    def normality_analysis(self, data_orig, data_xform,  xform):

        '''
        Method performs data transformations on the dataset.
        The method returns the transformed dataset and a dataframe
        containing the transformation statistics.
        
        Parameters:
        -----------
        self: The TransformationAnalysis object
        data_orig: The dataset before the transformation
        data_xform: The dataset after the transformation
        xform: The transformation used on the data

        Returns:
        --------
        df_summary: dataframe
        '''

        # Perform Shapiro-Wilk Test
        _, sw_orig = stats.shapiro(data_orig)
        _, sw_xform = stats.shapiro(data_xform)

        # Perform D'Agostino's K^2 Test
        _, da_orig = normaltest(data_orig)
        _, da_xform = normaltest(data_xform)

        # Standardize the Data (z-scores)
        z_orig = (data_orig - np.mean(data_orig)) / np.std(data_orig)
        z_xform = (data_xform - np.mean(data_xform)) / np.std(data_xform)

        # Perform Kolmogorov-Smirnov Test
        _, ks_orig = stats.kstest(z_orig, 'norm')
        _, ks_xform = stats.kstest(z_xform, 'norm')

        # Perform Skewness Test
        skew_orig = stats.skew(data_orig)
        skew_xform = stats.skew(data_xform)

        # Perform Kurtosis Test
        kurt_orig = stats.kurtosis(data_orig)
        kurt_xform = stats.kurtosis(data_xform)

        # Determine Improvements
        sw_improve = 'Yes' if sw_xform > sw_orig*1.1 else 'No'
        da_improve = 'Yes' if da_xform > da_orig*1.1 else 'No'
        ks_improve = 'Yes' if ks_xform > ks_orig*1.1 else 'No'
        skew_improve = 'Yes' if abs(skew_xform) < abs(skew_orig)*0.9 else 'No'
        kurt_improve = 'Yes' if abs(kurt_xform) < abs(kurt_orig)*0.9 else 'No'

        # Create List of Metrics Used
        metrics = ['Shapiro-Wilk', "D'Agostino's K^2", 'Kolmogorov-Smirnov', 'Skewness', 'Kurtosis']
        
        # Create List Original Data Metrics
        orig_values = [sw_orig, da_orig, ks_orig, skew_orig, kurt_orig]

        # Create List Transformed Data Metrics
        xform_values = [sw_xform, da_xform, ks_xform, skew_xform, kurt_xform]

        # Create List of Improvment Values
        improvements = [sw_improve, da_improve, ks_improve, skew_improve, kurt_improve]

        # Create Summary DataFrame
        df_summary = pd.DataFrame({
            'Metric': metrics,
            'Original Data': orig_values,
            f'{xform}': xform_values,
            'Improvement': improvements
        })

        # Return Results
        return df_summary






    ############################## Analysis Methods ############################



    # Define Method for Best Transformation
    def best_transformation(self, dataframe, data):

        '''
        Method analyzes the transformation statistics to determine the 
        best transformation method.
        
        Parameters:
        -----------
        self: The TransformationAnalysis object
        dataframe: A dataframe with the transformation stats
        data: A list of the transformed datasets

        Returns:
        --------
        results: list
        '''

        # Create Lists from DataFrame Column Values
        transformations = dataframe['Transformation'].tolist()
        skewness_values = dataframe['Skewness'].tolist()
        kurtosis_values = dataframe['Kurtosis'].tolist()

        # Replace nan Values with None
        skewness_values = [None if np.isnan(x) else x for x in skewness_values]

        # Find Valid Indices in Skewness Values
        valid_indices = [i for i, x in enumerate(skewness_values) if x is not None]

        # Calculate Transformation with Lowest Skewness
        best_index = valid_indices[np.argmin([abs(skewness_values[i]) for i in valid_indices])]

        # Set Transformation Variables
        xform_name = transformations[best_index]
        xform_data = data[best_index]
        xform_skew = skewness_values[best_index]
        xform_kurt = kurtosis_values[best_index]
        
        # Create Results List
        results = [xform_name, xform_data, xform_skew, xform_kurt]

        # Return Results List
        return results
    



    # Define Method for Improvement Analysis
    def improvement_analysis(self, dataframe, xform):

        '''
        Method analyzes the transformation statistics and
        prints the results.
        
        Parameters:
        -----------
        self: The TransformationAnalysis object
        dataframe: A dataframe with the transformation stats
        xform: The transformation used on the data

        Returns:
        --------
        df_improvements: dtdaframe

        Outputs
        -------
        None
        '''

        # Create Lists from DataFrame Column Values
        xform_values  = dataframe[xform].tolist()
        improvements = dataframe['Improvement'].tolist()

        # Set Transformation Variables
        sw_xform = xform_values[0]
        skew_xform = xform_values[3]
        kurt_xform = xform_values[4]

        # Calculate Improvement Scores
        total_improvements = 0
        total_metrics = len(improvements)

        for improvement in improvements:
            if improvement == 'Yes':
                total_improvements += 1


        # Determine Improvements Results
        i_interpretation = f'{total_improvements} out of {total_metrics} metrics show improvement'


        # Determine Transformation Results
        if total_improvements >= total_metrics/2:
            t_interpretation = 'Transformation improved normality'
        else:
            t_interpretation = 'Transformation did not substantially improve normality'


        # Determine Final Conclusion
        if total_improvements >= total_metrics/2:
            if sw_xform > 0.05 or (abs(skew_xform) < 0.5 and abs(kurt_xform) < 0.5):
                c_interpretation = f"The {xform} transformation significantly improved the dataset's normality"
            else:
                c_interpretation = f'The {xform} transformation helped but did not fully normalize the dataset'
        else:
            c_interpretation = 'Another transformation technique may be required'


        # Create Dataframe for Improvements Results
        df_improvements = pd.DataFrame({
            'Metric': ['Analysis Results', 'Transformation Results', 'Final Conclusion'],
            'Result': [i_interpretation, t_interpretation, c_interpretation]
        })

        # Return Results
        return df_improvements




    ############################## Print Methods ###############################



    # Define Methon for Printing Results
    def print_results(self, summary_data, results_data, stats_data, improvements_data, group, feature):

        '''
        Method prints the statistical results of the transformation analysis.
        
        Parameters:
        -----------
        self: The TransformationAnalysis object
        summary_data: The summary dataframe to print
        results_data: The results dataframe to print
        improvements_data: The improvements dataframe to print
        feature: The column to measure

        Returns:
        --------
        None

        Outputs:
        --------
        print statements
        summary table
        results table
        '''

        # Copy Dataframes
        df_summary_data = summary_data.copy()
        df_results_data = results_data.copy()
        df_stats_data = stats_data.copy()
        df_improvements_data = improvements_data.copy()

        # Create Title for Labels
        feature_title = self.titlize_column_name(feature)

        # Create Stats Column Labels
        column_list = stats_data.columns.tolist()

        # Set Transformation Variable
        transformation = column_list[2]

        # Format Columns to Six Decimal Places
        df_summary_data['Skewness'] = df_summary_data['Skewness'].apply(lambda x: f'{x:.6f}')
        df_summary_data['Kurtosis'] = df_summary_data['Kurtosis'].apply(lambda x: f'{x:.6f}')
        df_stats_data['Original Data'] = df_stats_data['Original Data'].apply(lambda x: f'{x:.6f}')
        df_stats_data[transformation] = df_stats_data[transformation].apply(lambda x: f'{x:.6f}')

        # Print Summary Data
        summary_title = f'Summary of Transformation Analysis for {feature_title} in {group}'
        summary_col_widths = [0.4, 0.3, 0.3]
        self.plot_table_data(df_summary_data, summary_title, summary_col_widths, 'left')

        # Print Transformation Results
        results_title = f'Transformation Results for Normalizing {feature_title} in {group}'
        results_col_widths = [0.4, 0.6]
        self.plot_table_data(df_results_data, results_title, results_col_widths, 'left')

        # Print Statistical Data
        stats_title = f'Normality Metrics Comparison for {feature_title} in {group}'
        stats_col_widths = [0.25, 0.25, 0.25, 0.25]
        self.plot_table_data(df_stats_data, stats_title, stats_col_widths, 'left')

        # Print Improvements Data
        improvements_title = f'Overall Assessment of Transformation Analysis on {feature_title} in {group}'
        improvements_col_widths = [0.3, 0.7]
        self.plot_table_data(df_improvements_data, improvements_title, improvements_col_widths, 'left')



    ############################## Run Methods #################################

    
    
    # Define Method for Transformation Analysis
    def transformation_analysis(self, data, group, feature, unit_test):

        '''
        Method performs data transformations on the data and
        prints graphs and tables of the results.
        
        Parameters:
        -----------
        self: The TransformationAnalysis object
        data: The dataset for the transformation
        group: The subgroup for the analysis
        feature: The columns to measure
        unit_test: Determines if unit tests are being performed

        Returns:
        --------
        analysis_results: list

        Outputs:
        --------
        Figure
        histogram plots
        Q-Q plots
        tables
        '''


        # Original Data
        data_orig = data[feature].copy()
        skew_orig = stats.skew(data_orig)


        # Perform Data Transformations
        df_summary, data_list = self.data_transformations(data, feature)

        
        # Determine Best Transformation
        xform, data_xform, skew_xform, kurt_xform = self.best_transformation(df_summary, data_list)

        # Set Group Variables
        group_orig = 'Original Data'
        group_xform = f'{xform} Transformation'

        # Plot Data
        if unit_test == False:
            self.plot_transformations(data_orig, group, feature, group_orig)
            self.plot_transformations(data_xform, group, feature, group_xform)

        # Create Dataframe for Transformation Results
        df_results = pd.DataFrame({
            'Metric': ['Best Transformation', 'Improvement'],
            'Result': [f'{xform}', f'Skewness reduced from {skew_orig:.4f} to {skew_xform:.4f}']
        })

        # Perform Normality Analysis
        df_stats = self.normality_analysis(data_orig, data_xform, xform)

        # Perform Improvement Analysis
        df_improvements = self.improvement_analysis(df_stats, xform)

        # Create Title for Labels
        feature_title = self.titlize_column_name(feature)

        # Create Summary Table
        summary_table = self.create_table(df_summary)
        
        # Print Transformation Analysis Results
        if unit_test == False:
            self.print_results(df_summary, df_results, df_stats, df_improvements, group, feature)

        # Create Results List
        analysis_results = [df_summary, df_results, df_stats, df_improvements]

        # Return Results List
        return analysis_results



   
    ############################## Main Method #################################



    # Define Methon to Apply Transformation Analysis
    def apply_transformation_analysis(self, column=None, feature=None, group=None, unit_test=None):

        '''
        Method performs transformation analysis on all groups in the selected column.
        
        Parameters:
        -----------
        self: The TransformationAnalysis object
        column: Column for subgroups
        feature: The columns to measure
        group: The subgroup for the analysis
        unit_test: Determines if unit tests are being performed

        Returns:
        --------
        transformation_results: list

        Outputs:
        --------
        Figure
        histogram plots
        Q-Q plots
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
            #print(f'\nAnalysis of Transformations in {group_1} data\n')
            print()

            # Filter Data
            filtered_data = data[data.apply(lambda row: row[column] == group_1, axis=1)]

            # Perform Transformation Analysis
            transformation_results = self.transformation_analysis(filtered_data, group_1, feature, unit_test)
        
        elif group == "second":

            # Print Group
            #print(f'\nAnalysis of Transformations in {group_2} data\n')
            print()

            # Filter Data
            filtered_data = data[data.apply(lambda row: row[column] == group_2, axis=1)]

            # Perform Tansformation Analysis
            transformation_results = self.transformation_analysis(filtered_data, group_2, feature, unit_test)

        else:
            # Iterate Through Each Group in Group List
            for subgroup in group_list:

                # Create Transformation Results List
                transformation_results = []

                # Print Group
                print(f'\nAnalysis of Transformations in {subgroup}\n')

                # Filter Data
                filtered_data = data[data.apply(lambda row: row[column] == subgroup, axis=1)]

                # Perform Transformation Analysis
                analysis_results = self.transformation_analysis(filtered_data, group, feature, unit_test)

                # Append List
                transformation_results.append(analysis_results)
        
        # Return Transformation Results
        return transformation_results















# End of Page
