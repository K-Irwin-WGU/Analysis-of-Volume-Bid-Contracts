'''data_analysis_main'''


# IMPORT PACKAGES

import numpy as np
import pandas as pd

from process_data import ProcessData
from visual_analysis import VisualAnalysis
from outlier_analysis import OutlierAnalysis
from distribution_analysis import DistributionAnalysis
from transformation_analysis import TransformationAnalysis
from t_test_analysis import TTestAnalysis
from u_test_analysis import UTestAnalysis
from parametric_nonparametric_analysis import ParametricNonparametricAnalysis
from chi_square_analysis import ChiSquareAnalysis




# CREATE ANALYSIS MAIN CLASS

class AnalysisMain():

    '''
    Class creates nine data anlysis objects, calls the objects methods,
    and returns the results of the methods of each object.

    Returns:
    --------

    Process Data Class Returns
    df_loaded_data: dataframe
    loaded_data_columns = list
    df_processed_data: dataframe

    Outlier Analysis Class Returns
    outlier_results_1:  series/list
    outlier_results_2:  series/list
    outlier_results_3:  series/list
    outlier_results_4:  series/list

    Distribution Analysis Class Returns
    distribution_results_1: list
    distribution_results_2: list
    distribution_results_3: list
    distribution_results_4: list

    Transformation Analysis Class Returns
    transformation_results_1: list
    transformation_results_2: list
    transformation_results_3: list
    transformation_results_4: list

    T-Test Analysis Analysis Class Returns
    t_test_results_1: dictionary
    t_test_results_2: dictionary
    t_test_results_3: dictionary
    t_test_results_4: dictionary

    U-Test Analysis Analysis Class VReturns
    u_test_results_1: dictionary
    u_test_results_2: dictionary

    Parametric-Nonparametric Analysis Analysis Class Returns
    parametric_nonparametric_results_1: dictionary
    parametric_nonparametric_results_2: dictionary

    Chi-Square Analysis Analysis ClassReturns
    chi_squared_results_1: dictionary
    '''

    # Define init Method
    def __init__(self):

        '''
        Method initializes class variables
        and creates data analysis objects.
        
        Parameters:
        -----------
        self: The AnalysisMain object

        Outputs:
        --------
        self.pd: ProcessData object
        self.va: VisualAnalysis object
        self.oa: OutlierAnalysis object
        self.da: DistributionAnalysis object
        self.ta: TransformationAnalysis object
        self.tt: TTestAnalysis object
        self.ut: UTestAnalysis object
        self.pnp: ParametricNonparametricAnalysis object
        self.cs: ChiSquareAnalysis object
        '''

        # Initialize Class Variables for ProcessData Object
        self._df_loaded_data = None
        self._loaded_data_columns = None
        self._df_processed_data = None

        # Initialize Class Variables for OutlierAnalysis Object
        self._outlier_results_1 = None
        self._outlier_results_2 = None
        self._outlier_results_3 = None
        self._outlier_results_4 = None

        # Initialize Class Variables for DistributionAnalysis Object
        self._distribution_results_1 = None
        self._distribution_results_2 = None
        self._distribution_results_3 = None
        self._distribution_results_4 = None

        # Initialize Class Variables for TransformationAnalysis Object
        self._transformation_results_1 = None
        self._transformation_results_2 = None
        self._transformation_results_3 = None
        self._transformation_results_4 = None

        # Initialize Class Variables for TTestAnalysis Object
        self._t_test_results_1 = None
        self._t_test_results_2 = None
        self._t_test_results_3 = None
        self._t_test_results_4 = None

        # Initialize Class Variables for UTestAnalysis Object
        self._u_test_results_1 = None
        self._u_test_results_2 = None

        # Initialize Class Variables for ParametricNonparametricAnalysis Object
        self._parametric_nonparametric_results_1 = None
        self._parametric_nonparametric_results_2 = None

        # Initialize Class Variables for ChiSquareAnalysis Object
        self._chi_squared_results_1 = None

        # Create Data Analysis Objects
        self.pd = ProcessData()
        self.va = VisualAnalysis()
        self.oa = OutlierAnalysis()
        self.da = DistributionAnalysis()
        self.ta = TransformationAnalysis()
        self.tt = TTestAnalysis()
        self.ut = UTestAnalysis()
        self.pnp = ParametricNonparametricAnalysis()
        self.cs = ChiSquareAnalysis()




    ############################## Getter Methods ##############################



    '''
    Methods get values from each Process Data class variable.
    '''

    # Getter for Loaded Data
    @property
    def df_loaded_data(self):  
        return self._df_loaded_data


    # Getter for Loaded Data Columns
    @property
    def loaded_data_columns(self):  
        return self._loaded_data_columns
    

    # Getter for Processed Data
    @property
    def df_processed_data(self):  
        return self._df_processed_data
    



    '''
    Methods get values from each Outlier Analysis class variable.
    '''

    # Getter for Outlier Results 1
    @property
    def outlier_results_1(self):  
        return self._outlier_results_1


    # Getter for Outlier Results 2
    @property
    def outlier_results_2(self):  
        return self._outlier_results_2
    

    # Getter for Outlier Results 3
    @property
    def outlier_results_3(self):  
        return self._outlier_results_3
    

    # Getter for Outlier Results 4
    @property
    def outlier_results_4(self):  
        return self._outlier_results_4
    



    '''
    Methods get values from each Distribution Analysis class variable.
    '''

    # Getter for Distribution Results 1
    @property
    def distribution_results_1(self):  
        return self._distribution_results_1


    # Getter for Distribution Results 2
    @property
    def distribution_results_2(self):  
        return self._distribution_results_2
    

    # Getter for Distribution Results 3
    @property
    def distribution_results_3(self):  
        return self._distribution_results_3
    

    # Getter for Distribution Results 4
    @property
    def distribution_results_4(self):  
        return self._distribution_results_4
    



    '''
    Methods get values from each TransformationAnalysis Analysis class variable.
    '''

    # Getter for Transformation Results 1
    @property
    def transformation_results_1(self):  
        return self._transformation_results_1


    # Getter for Transformation Results 2
    @property
    def transformation_results_2(self):  
        return self._transformation_results_2
    

    # Getter for Transformation Results 3
    @property
    def transformation_results_3(self):  
        return self._transformation_results_3
    

    # Getter for Transformation Results 4
    @property
    def transformation_results_4(self):  
        return self._transformation_results_4
    



    '''
    Methods get values from each T-Test Analysis class variable.
    '''

    # Getter for T-Test Results 1
    @property
    def t_test_results_1(self):  
        return self._t_test_results_1


    # Getter for T-Test Results 2
    @property
    def t_test_results_2(self):  
        return self._t_test_results_2
    

    # Getter for T-Test Results 3
    @property
    def t_test_results_3(self):  
        return self._t_test_results_3


    # Getter for T-Test Results 4
    @property
    def t_test_results_4(self):  
        return self._t_test_results_4
    



    '''
    Methods get values from each U-Test Analysis class variable.
    '''

    # Getter for U-Test Results 1
    @property
    def u_test_results_1(self):  
        return self._u_test_results_1


    # Getter for U-Test Results 2
    @property
    def u_test_results_2(self):  
        return self._u_test_results_2
    



    '''
    Methods get values from each Parametric Nonparametric Analysis class variable.
    '''

    # Getter for Parametric Nonparametric Results 1
    @property
    def parametric_nonparametric_results_1(self):  
        return self._parametric_nonparametric_results_1


    # Getter for Parametric Nonparametric Results 2
    @property
    def parametric_nonparametric_results_2(self):  
        return self._parametric_nonparametric_results_2
    



    '''
    Methods get values from each Chi-Square Analysis class variable.
    '''

    # Getter for Chi-Square Results 1
    @property
    def chi_squared_results_1(self):  
        return self._chi_squared_results_1




    ############################## Setter Methods ##############################



    '''
    Methods set values for each Process Data class variable.
    '''

    # Define Setter for Loaded Data Variable
    @df_loaded_data.setter
    def df_loaded_data(self, dataframe):
        if isinstance(dataframe, pd.DataFrame):
            self._df_loaded_data = dataframe
        else:
            raise ValueError('Data must be a DataFrame.')




    # Define Setter for Load Data Columns Variable
    @loaded_data_columns.setter
    def loaded_data_columns(self, data_list):
        if isinstance(data_list, list) and len(data_list) > 0:
            self._loaded_data_columns = data_list
        else:
            raise ValueError('Data list must be a non-empty list.')
    



    # Define Setter for Processed Data Variable
    @df_processed_data.setter
    def df_processed_data(self, dataframe):
        if isinstance(dataframe, pd.DataFrame):
            self._df_processed_data = dataframe
        else:
            raise ValueError('Data must be a DataFrame.')
    



    '''
    Methods set values for each Outlier Analysis class variable.
    '''

    # Define Setter for Outlier Results 1 Variable
    @outlier_results_1.setter
    def outlier_results_1(self, data):
        if isinstance(data, (list, pd.Series)):
            self._outlier_results_1 = data
        else:
            raise ValueError('Data must be a Series or list.')
    



    # Define Setter for Outlier Results 2 Variable
    @outlier_results_2.setter
    def outlier_results_2(self, data):
        if isinstance(data, (list, pd.Series)):
            self._outlier_results_2 = data
        else:
            raise ValueError('Data must be a Series or list.')
    



    # Define Setter for Outlier Results 3 Variable
    @outlier_results_3.setter
    def outlier_results_3(self, data):
        if isinstance(data, (list, pd.Series)):
            self._outlier_results_3 = data
        else:
            raise ValueError('Data must be a Series or list.')
    



    # Define Setter for Outlier Results 4 Variable
    @outlier_results_4.setter
    def outlier_results_4(self, data):
        if isinstance(data, (list, pd.Series)):
            self._outlier_results_4 = data
        else:
            raise ValueError('Data must be a Series or list.')
    



    '''
    Methods set values for each Distribution Analysis class variable.
    '''

    # Define Setter for Distribution Results 1 Variable
    @distribution_results_1.setter
    def distribution_results_1(self, data_list):
        if isinstance(data_list, list) and len(data_list) > 0:
            self._distribution_results_1 = data_list
        else:
            raise ValueError('Data list must be a non-empty list.')
    



    # Define Setter for Distribution Results 2 Variable
    @distribution_results_2.setter
    def distribution_results_2(self, data_list):
        if isinstance(data_list, list) and len(data_list) > 0:
            self._distribution_results_2 = data_list
        else:
            raise ValueError('Data list must be a non-empty list.')
    



    # Define Setter for Distribution Results 3 Variable
    @distribution_results_3.setter
    def distribution_results_3(self, data_list):
        if isinstance(data_list, list) and len(data_list) > 0:
            self._distribution_results_3 = data_list
        else:
            raise ValueError('Data list must be a non-empty list.')
    



    # Define Setter for Distribution Results 4 Variable
    @distribution_results_4.setter
    def distribution_results_4(self, data_list):
        if isinstance(data_list, list) and len(data_list) > 0:
            self._distribution_results_4 = data_list
        else:
            raise ValueError('Data list must be a non-empty list.')
    



    '''
    Methods set values for each Transformation Analysis class variable.
    '''

    # Define Setter for Transformation Results 1 Variable
    @transformation_results_1.setter
    def transformation_results_1(self, data_list):
        if isinstance(data_list, list) and len(data_list) > 0:
            self._transformation_results_1 = data_list
        else:
            raise ValueError('Data list must be a non-empty list.')
    



    # Define Setter for Transformation Results 2 Variable
    @transformation_results_2.setter
    def transformation_results_2(self, data_list):
        if isinstance(data_list, list) and len(data_list) > 0:
            self._transformation_results_2 = data_list
        else:
            raise ValueError('Data list must be a non-empty list.')
    



    # Define Setter for Transformation Results 3 Variable
    @transformation_results_3.setter
    def transformation_results_3(self, data_list):
        if isinstance(data_list, list) and len(data_list) > 0:
            self._transformation_results_3 = data_list
        else:
            raise ValueError('Data list must be a non-empty list.')
    



    # Define Setter for Transformation Results 4 Variable
    @transformation_results_4.setter
    def transformation_results_4(self, data_list):
        if isinstance(data_list, list) and len(data_list) > 0:
            self._transformation_results_4 = data_list
        else:
            raise ValueError('Data list must be a non-empty list.')
    



    '''
    Methods set values for each T-Test Analysis class variable.
    '''

    # Define Setter for T-Test Results 1 Variable
    @t_test_results_1.setter
    def t_test_results_1(self, data_dict):
        if isinstance(data_dict, dict) and len(data_dict) > 0:
            self._t_test_results_1 = data_dict
        else:
            raise ValueError('Data must be a non-empty dictionary')
    



    # Define Setter for T-Test Results 2 Variable
    @t_test_results_2.setter
    def t_test_results_2(self, data_dict):
        if isinstance(data_dict, dict) and len(data_dict) > 0:
            self._t_test_results_2 = data_dict
        else:
            raise ValueError('Data must be a non-empty dictionary.')
    



     # Define Setter for T-Test Results 3 Variable
    @t_test_results_3.setter
    def t_test_results_3(self, data_dict):
        if isinstance(data_dict, dict) and len(data_dict) > 0:
            self._t_test_results_3 = data_dict
        else:
            raise ValueError('Data must be a non-empty dictionary')
    



    # Define Setter for T-Test Results 4 Variable
    @t_test_results_4.setter
    def t_test_results_4(self, data_dict):
        if isinstance(data_dict, dict) and len(data_dict) > 0:
            self._t_test_results_4 = data_dict
        else:
            raise ValueError('Data must be a non-empty dictionary.')
    



    '''
    Methods set values for each U-Test Analysis class variable.
    '''

    # Define Setter for U-Test Results 1 Variable
    @u_test_results_1.setter
    def u_test_results_1(self, data_dict):
        if isinstance(data_dict, dict) and len(data_dict) > 0:
            self._u_test_results_1 = data_dict
        else:
            raise ValueError('Data must be a non-empty dictionary')
    



    # Define Setter for U-Test Results 2 Variable
    @u_test_results_2.setter
    def u_test_results_2(self, data_dict):
        if isinstance(data_dict, dict) and len(data_dict) > 0:
            self._u_test_results_2 = data_dict
        else:
            raise ValueError('Data must be a non-empty dictionary.')
    



    '''
    Methods set values for each Parametric Nonparametric Analysis class variable.
    '''

    # Define Setter for Parametric Nonparametric Results 1 Variable
    @parametric_nonparametric_results_1.setter
    def parametric_nonparametric_results_1(self, data_dict):
        if isinstance(data_dict, dict) and len(data_dict) > 0:
            self._parametric_nonparametric_results_1 = data_dict
        else:
            raise ValueError('Data must be a non-empty dictionary')
    



    # Define Setter for Parametric Nonparametric Results 2 Variable
    @parametric_nonparametric_results_2.setter
    def parametric_nonparametric_results_2(self, data_dict):
        if isinstance(data_dict, dict) and len(data_dict) > 0:
            self._parametric_nonparametric_results_2 = data_dict
        else:
            raise ValueError('Data must be a non-empty dictionary.')
    



    '''
    Methods set values for each Chi-Square Analysis class variable.
    '''

    # Define Setter for Chi-Square Results 1 Variable
    @chi_squared_results_1.setter
    def chi_squared_results_1(self, data_dict):
        if isinstance(data_dict, dict) and len(data_dict) > 0:
            self._chi_squared_results_1 = data_dict
        else:
            raise ValueError('Data must be a non-empty dictionary')




    ############################## Process Data Methods ##########################################



    # Define Method to Manually Set Process Data Variables
    def pd_set_manual(self, data_file_1=None, data_file_2=None, 
                data_file_3=None, folder_1=None, folder_2=None, 
                dataset_1=None, dataset_2=None, dataset_3=None):
        
        '''
        Method manually sets class variables for the ProcessData object.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        data_file_1: The name of the first CSV file to be loaded
        data_file_2: The name of the second CSV file to be loaded
        data_file_3: The name of the CSV file to be saved
        folder_1: The name of the folder containing the CSV files to be loaded
        folder_2: The name of the folder containing the CSV files to be saved
        dataset_1: The name of the first dataset to be loaded
        dataset_2: The name of the second dataset to be loaded
        dataset_3: The name of the dataset to be saved

        Returns:
        --------
        None

        Outputs:
        --------
        None
        '''

        self.pd.data_file_1 = data_file_1
        self.pd.data_file_2 = data_file_2
        self.pd.data_file_3 = data_file_3
        self.pd.folder_1 = folder_1
        self.pd.folder_2 = folder_2
        self.pd.dataset_1 = dataset_1
        self.pd.dataset_2 = dataset_2
        self.pd.dataset_3 = dataset_3




    # Define Method to Automatically Set Process Data Variables
    def pd_set_automatically(self):

        '''
        Method automatically sets class variables for the ProcessData object.
        
        Parameters:
        -----------
        self: The AnalysisMain object

        Returns:
        --------
        None

        Outputs:
        --------
        None
        '''

        self.pd.data_file_1 = 'Work_Completed_Set1.csv'
        self.pd.data_file_2 = 'Work_Completed_Set2.csv'
        self.pd.data_file_3 = 'Work_Completed_Data.csv'
        self.pd.folder_1 = 'Clean-Data'
        self.pd.folder_2 = 'Processed-Data'
        self.pd.dataset_1 = 'Historic Data'
        self.pd.dataset_2 = 'Current Data'
        self.pd.dataset_3 = 'Combined Data'




    # Define Method to Load Data
    def load_data(self, data_file_1=None, data_file_2=None, 
                data_file_3=None, folder_1=None, folder_2=None, 
                dataset_1=None, dataset_2=None, dataset_3=None, auto_set=True):
        
        '''
        Mehton calls the load_data method from the ProcessData class.
        The method loads data from CSV files and prints out tables
        with column information for each dataset.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        data_file_1: The name of the first CSV file to be loaded
        data_file_2: The name of the second CSV file to be loaded
        data_file_3: The name of the CSV file to be saved
        folder_1: The name of the folder containing the CSV files to be loaded
        folder_2: The name of the folder containing the CSV files to be saved
        dataset_1: The name of the first dataset to be loaded
        dataset_2: The name of the second dataset to be loaded
        dataset_3: The name of the dataset to be saved
        auto_set: Determines if object variables will be set automatically

        Returns:
        --------
        df_loaded_data: dataframe
        loaded_data_columns: list

        Outputs:
        --------
        figure
        table 1
        table 2
        '''

        # Set Process Data Variables
        if auto_set == True:
            self.pd_set_automatically()
        else:
            self.pd_set_manual(data_file_1, data_file_2, data_file_3, folder_1, 
                                folder_2, dataset_1, dataset_2, dataset_3)
        
        # Load Data
        df_loaded_data, loaded_data_columns = self.pd.load_data()

        # Set Class Variables
        self.df_loaded_data = df_loaded_data
        self.loaded_data_columns = loaded_data_columns




    # Define Method to Process Data
    def process_data(self):

        '''
        Mehton calls the process_data method from the ProcessData class.
        The method processes the loaded data for analysis and
        prints out a table with the column information.
        
        Parameters:
        -----------
        self: The AnalysisMain object

        Returns:
        --------
        df_processed_data: dataframe

        Outputs:
        --------
        figure
        table
        '''

        # Set Data Variables
        df_loaded_data = self.df_loaded_data
        loaded_data_columns = self.loaded_data_columns
        
        # Process Data
        df_processed_data = self.pd.process_data(df_loaded_data, loaded_data_columns)

        # Set Class Variables
        self.df_processed_data = df_processed_data




    ############################## Visual Analysis Methods #######################################



    # Define Method to Manually Set Visual Analysis Variables
    def va_set_manual(self, data=None, column=None, features=None, 
                      feature_1=None, feature_2=None, target=None, target_slice=None):
        
        '''
        Method manually sets class variables for the VisualAnalysis object.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        data: Dataset to analyze
        column: Column for subgroups
        features: list of columns to measure
        feature_1: The first column to measure
        feature_2: The second column to measure
        target: Column of the target of comparison
        target_slice: Number of charectors to include in target value

        Returns:
        --------
        None

        Outputs:
        --------
        None
        '''

        # Set Object Variables
        self.va.data = data
        self.va.column = column
        self.va.features = features
        self.va.feature_1 = feature_1
        self.va.feature_2 = feature_2
        self.va.target = target
        self.va.target_slice = target_slice




    # Define Method to Automatically Set Visual Analysis Variables
    def va_set_automatically(self):

        '''
        Method automatically sets class variables for the VisualAnalysis object.
        
        Parameters:
        -----------
        self: The AnalysisMain object

        Returns:
        --------
        None

        Outputs:
        --------
        None
        '''

        # Set Object Variables
        self.va.data = self.df_processed_data
        self.va.column = 'dataset_name'
        self.va.features = ['hours_to_complete', 'days_past_due']
        self.va.feature_1 = 'hours_to_complete'
        self.va.feature_2 = 'days_past_due'
        self.va.target = 'priority'
        self.va.target_slice = None
    



    # Define Method for Histogram Plot Analysis
    def histogram_plot_analysis(self, data=None, column=None, features=None, auto_set=True):

        '''
        Mehton calls the histogram_plot_analysis method from the VisualAnalysis class.
        The method creates a series of histograms displaying measures of central tendency
        for the selected column and features.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        data: Dataset to analyze
        column: Column for subgroups
        features: list of columns to measure
        auto_set: Determines if object variables will be set automatically

        Returns:
        --------
        None

        Outputs:
        --------
        figure
        histogram plots
        '''

        # Set Visual Analysis Variables
        if auto_set == True:
            self.va_set_automatically()
        else:
            self.va_set_manual(data=data, column=column, features=features)
        
        # Perform Histogram Plot Analysis
        self.va.histogram_plot_analysis()
    



    # Define Method for Box Plot Analysis
    def box_plot_analysis(self, data=None, column=None, features=None, auto_set=True):

        '''
        Mehton calls the box_plot_analysis method from the VisualAnalysis class.
        The method creates a series of box plots displaying measures of central tendency
        and outliers for the selected column and features.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        data: Dataset to analyze
        column: Column for subgroups
        features: list of columns to measure
        auto_set: Determines if object variables will be set automatically

        Returns:
        --------
        None

        Outputs:
        --------
        figure
        box plots
        '''

        # Set Visual Analysis Variables
        if auto_set == True:
            self.va_set_automatically()
        else:
            self.va_set_manual(data=data, column=column, features=features)
        
        # Perform Box Plot Analysis
        self.va.box_plot_analysis()
    



    # Define Method for Count Plot Analysis 1
    def count_plot_analysis_1(self, data=None, column=None, target=None, auto_set=True):

        '''
        Mehton calls the count_plot_analysis method from the VisualAnalysis class.
        The method creates a count plot of subgroups comparing a target status.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        data: Dataset to analyze
        column: Column for subgroups
        target: Column of the target of comparison
        auto_set: Determines if object variables will be set automatically

        Returns:
        --------
        None

        Outputs:
        --------
        figure
        count plot
        '''

        # Set Visual Analysis Variables
        if auto_set == True:
            self.va_set_automatically()
            self.va.target = 'priority'
            self.va.target_slice = 2
        else:
            self.va_set_manual(data=data, column=column, target=target)
        
        # Perform Count Plot Analysis
        self.va.count_plot_analysis()
    



    # Define Method for Count Plot Analysis 2
    def count_plot_analysis_2(self, data=None, column=None, target=None, auto_set=True):

        '''
        Mehton calls the count_plot_analysis method from the VisualAnalysis class.
        The method creates a count plot of subgroups comparing a target status.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        data: Dataset to analyze
        column: Column for subgroups
        target: Column of the target of comparison
        auto_set: Determines if object variables will be set automatically

        Returns:
        --------
        None

        Outputs:
        --------
        figure
        count plot
        '''

        # Set Visual Analysis Variables
        if auto_set == True:
            self.va_set_automatically()
            self.va.target = 'sla_label'
        else:
            self.va_set_manual(data=data, column=column, target=target)
        
        # Perform Count Plot Analysis
        self.va.count_plot_analysis()
    



    # Define Method for Violin Plot Analysis 1
    def violin_plot_analysis_1(self, data=None, column=None, feature=None, target=None, auto_set=True):

        '''
        Mehton calls the violin_plot_analysis method from the VisualAnalysis class.
        The method creates violin plots comparing subgroups for each target value.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The column to measure
        target: Column of the target of comparison
        auto_set: Determines if object variables will be set automatically

        Returns:
        --------
        None

        Outputs:
        --------
        figure
        violin plots
        '''

        # Set Visual Analysis Variables
        if auto_set == True:
            self.va_set_automatically()
            self.va.feature_1 = 'hours_to_complete'
        else:
            self.va_set_manual(data=data, column=column, feature_1=feature, target=target)
        
        # Perform Violin Plot Analysis
        self.va.violin_plot_analysis()
    



    # Define Method for Violin Plot Analysis 2
    def violin_plot_analysis_2(self, data=None, column=None, feature=None, target=None, auto_set=True):

        '''
        Mehton calls the violin_plot_analysis method from the VisualAnalysis class.
        The method creates violin plots comparing subgroups for each target value.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The column to measure
        target: Column of the target of comparison
        auto_set: Determines if object variables will be set automatically

        Returns:
        --------
        None

        Outputs:
        --------
        figure
        violin plots
        '''

        # Set Visual Analysis Variables
        if auto_set == True:
            self.va_set_automatically()
            self.va.feature_1 = 'days_past_due'
        else:
            self.va_set_manual(data=data, column=column, feature_1=feature, target=target)
        
        # Perform Violin Plot Analysis
        self.va.violin_plot_analysis()
    



    # Define Method for KDE Plot Analysis
    def kde_plot_analysis(self, data=None, column=None, features=None, target=None, auto_set=True):

        '''
        Mehton calls the kde_plot_analysis method from the VisualAnalysis class.
        The method visualizes main data groups and targets by key features
        and displays differences in means.

        Parameters:
        -----------
        self: The AnalysisMain object
        data: Dataset to be analyzed
        column: The column by which the data is segmented
        features: The list of columns by which the values are measured
        target: The column by which the segments are compared
        auto_set: Determines if object variables will be set automatically

        Returns:
        --------
        None

        Outputs:
        --------
        figure
        kde plots
        '''

        # Set Visual Analysis Variables
        if auto_set == True:
            self.va_set_automatically()
            self.va.target = 'sla_label'
        else:
            self.va_set_manual(data=data, column=column, features=features, target=target)
        
        # Perform KDE Plot Analysis
        self.va.kde_plot_analysis()
    



    # Define a Method for Statistical Summary 1
    def statistic_summary_1(self, data=None, column=None, feature=None, target=None, auto_set=True):

        '''
        Mehton calls the statistic_summary method from the VisualAnalysis class.
        The method creates summary statistics for each subgroup and each target value.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The column to measure
        target: Column of the target of comparison
        auto_set: Determines if object variables will be set automatically

        Outputs:
        --------
        figure
        table
        '''

        # Set Visual Analysis Variables
        if auto_set == True:
            self.va_set_automatically()
            self.va.feature_1 = 'hours_to_complete'
            self.va.target = 'sla_label'
        else:
            self.va_set_manual(data=data, column=column, feature_1=feature, target=target)
        
        # Perform Statistical Summary Analysis
        self.va.statistic_summary()
    



    # Define a Method for Statistical Summary 2
    def statistic_summary_2(self, data=None, column=None, feature=None, target=None, auto_set=True):

        '''
        Mehton calls the statistic_summary method from the VisualAnalysis class.
        The method creates summary statistics for each subgroup and each target value.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The column to measure
        target: Column of the target of comparison
        auto_set: Determines if object variables will be set automatically

        Outputs:
        --------
        figure
        table
        '''

        # Set Visual Analysis Variables
        if auto_set == True:
            self.va_set_automatically()
            self.va.feature_1 = 'days_past_due'
            self.va.target = 'sla_label'
        else:
            self.va_set_manual(data=data, column=column, feature_1=feature, target=target)
        
        # Perform Statistical Summary Analysis
        self.va.statistic_summary()




    ############################## Outlier Analysis Methods ######################################



    # Define Method to Manually Set Outlier Analysis Variables
    def oa_set_manual(self, data=None, column=None, feature=None, group=None):
        
        '''
        Method manually sets class variables for the OutlierAnalysis object.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The columns to measure
        group: The subgroup for the analysis

        Returns:
        --------
        None

        Outputs:
        --------
        None
        '''
        
        # Set Object Variables
        self.oa.data = data
        self.oa.column = column
        self.oa.feature = feature
        self.oa.group = group
    



    # Define Method to Automatically Set Outlier Analysis Variables
    def oa_set_automatically(self):

        '''
        Method automatically sets class variables for the OutlierAnalysis object.
        
        Parameters:
        -----------
        self: The AnalysisMain object

        Returns:
        --------
        None

        Outputs:
        --------
        None
        '''

        # Set Object Variables
        self.oa.data = self.df_processed_data
        self.oa.column = 'dataset_name'
        self.oa.feature = 'hours_to_complete'
        self.oa.group = 'first'
        



    # Define Method for Outlier Analysis 1
    def outlier_analysis_1(self, data=None, column=None, feature=None, group=None, auto_set=True):

        '''
        Mehton calls the apply_outlier_analysis method from the OutlierAnalysis class.
        The method performs outlier analysis on all groups in the selected column.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        column: Column for subgroups
        feature: The columns to measure
        group: The subgroup for the analysis
        auto_set: Determines if object variables will be set automatically

        Returns:
        --------
        outliers_results: dataset/list

        Outputs:
        --------
        figure
        box plot
        histogram
        table
        '''

        # Set Outlier Analysis Variables
        if auto_set == True:
            self.oa_set_automatically()
        else:
            self.oa_set_manual(data=data, column=column, feature=feature, group=group)
        
        # Perform Outlier Analysis
        outlier_results = self.oa.apply_outlier_analysis()

        # Set Class Variables
        self.outlier_results_1 = outlier_results
    



    # Define Method for Outlier Analysis 2
    def outlier_analysis_2(self, data=None, column=None, feature=None, group=None, auto_set=True):

        '''
        Mehton calls the apply_outlier_analysis method from the OutlierAnalysis class.
        The method performs outlier analysis on all groups in the selected column.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        column: Column for subgroups
        feature: The columns to measure
        group: The subgroup for the analysis
        auto_set: Determines if object variables will be set automatically

        Returns:
        --------
        outliers_results: dataset/list

        Outputs:
        --------
        figure
        box plot
        histogram
        table
        '''

        # Set Outlier Analysis Variables
        if auto_set == True:
            self.oa_set_automatically()
            self.oa.group = 'second'
        else:
            self.oa_set_manual(data=data, column=column, feature=feature, group=group)
        
        # Perform Outlier Analysis
        outlier_results = self.oa.apply_outlier_analysis()

        # Set Class Variables
        self.outlier_results_2 = outlier_results
    



    # Define Method for Outlier Analysis 3
    def outlier_analysis_3(self, data=None, column=None, feature=None, group=None, auto_set=True):

        '''
        Mehton calls the apply_outlier_analysis method from the OutlierAnalysis class.
        The method performs outlier analysis on all groups in the selected column.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        column: Column for subgroups
        feature: The columns to measure
        group: The subgroup for the analysis
        auto_set: Determines if object variables will be set automatically

        Returns:
        --------
        outliers_results: dataset/list

        Outputs:
        --------
        figure
        box plot
        histogram
        table
        '''

        # Set Outlier Analysis Variables
        if auto_set == True:
            self.oa_set_automatically()
            self.oa.feature = 'days_past_due'
        else:
            self.oa_set_manual(data=data, column=column, feature=feature, group=group)
        
        # Perform Outlier Analysis
        outlier_results = self.oa.apply_outlier_analysis()

        # Set Class Variables
        self.outlier_results_3 = outlier_results
    



    # Define Method for Outlier Analysis 4
    def outlier_analysis_4(self, data=None, column=None, feature=None, group=None, auto_set=True):

        '''
        Mehton calls the apply_outlier_analysis method from the OutlierAnalysis class.
        The method performs outlier analysis on all groups in the selected column.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        column: Column for subgroups
        feature: The columns to measure
        group: The subgroup for the analysis
        auto_set: Determines if object variables will be set automatically

        Returns:
        --------
        outliers_results: dataset/list

        Outputs:
        --------
        figure
        box plot
        histogram
        table
        '''

        # Set Outlier Analysis Variables
        if auto_set == True:
            self.oa_set_automatically()
            self.oa.feature = 'days_past_due'
            self.oa.group = 'second'
        else:
            self.oa_set_manual(data=data, column=column, feature=feature, group=group)
        
        # Perform Outlier Analysis
        outlier_results = self.oa.apply_outlier_analysis()

        # Set Class Variables
        self.outlier_results_4 = outlier_results
    



    ############################## Distribution Analysis Methods #################################



    # Define Method to Manually Set Distribution Analysis Variables
    def da_set_manual(self, data=None, column=None, feature=None, target=None, group=None):
        
        '''
        Method manually sets class variables for the DistributionAnalysis object.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The columns to measure
        target: The column to subdivide data into groups
        group: The subgroup for the analysis

        Returns:
        --------
        None

        Outputs:
        --------
        None
        '''
        
        # Set Object Variables
        self.da.data = data
        self.da.column = column
        self.da.feature = feature
        self.da.target = target
        self.da.group = group
    



    # Define Method to Automatically Set Distribution Analysis Variables
    def da_set_automatically(self):

        '''
        Method automatically sets class variables for the DistributionAnalysis object.
        
        Parameters:
        -----------
        self: The AnalysisMain object

        Returns:
        --------
        None

        Outputs:
        --------
        None
        '''

        # Set Object Variables
        self.da.data = self.df_processed_data
        self.da.column = 'dataset_name'
        self.da.feature = 'hours_to_complete'
        self.da.target = 'priority'
        self.da.group = 'first'
        



    # Define Method for Distribution Analysis 1
    def distribution_analysis_1(self, data=None, column=None, feature=None, target=None, group=None, auto_set=True):

        '''
        Mehton calls the apply_outlier_analysis method from the DistributionAnalysis class.
        The method performs distribution analysis on a selected group in the selected column
        to determine its normality, skewness, and kurtosis.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        column: Column for subgroups
        feature: The columns to measure
        target: The column to subdivide data into groups
        group: The subgroup for the analysis
        auto_set: Determines if object variables will be set automatically

        Returns:
        --------
        distribution_results: dataset/list

        Outputs:
        --------
        figure
        histogram
        Q-Q plot
        box plot
        bar plot
        table
        '''

        # Set  Distribution Analysis Variables
        if auto_set == True:
            self.da_set_automatically()
        else:
            self.da_set_manual(data=data, column=column, feature=feature, target=target, group=group)
        
        # Perform  Distribution Analysis
        distribution_results = self.da.apply_distribution_analysis()

        # Set Class Variables
        self.distribution_results_1 = distribution_results
    



    # Define Method for Distribution Analysis 2
    def distribution_analysis_2(self, data=None, column=None, feature=None, target=None, group=None, auto_set=True):

        '''
        Mehton calls the apply_outlier_analysis method from the DistributionAnalysis class.
        The method performs distribution analysis on a selected group in the selected column
        to determine its normality, skewness, and kurtosis.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        column: Column for subgroups
        feature: The columns to measure
        target: The column to subdivide data into groups
        group: The subgroup for the analysis
        auto_set: Determines if object variables will be set automatically

        Returns:
        --------
        distribution_results: dataset/list

        Outputs:
        --------
        figure
        histogram
        Q-Q plot
        box plot
        bar plot
        table
        '''

        # Set  Distribution Analysis Variables
        if auto_set == True:
            self.da_set_automatically()
            self.da.group = 'second'
        else:
            self.da_set_manual(data=data, column=column, feature=feature, target=target, group=group)
        
        # Perform  Distribution Analysis
        distribution_results = self.da.apply_distribution_analysis()

        # Set Class Variables
        self.distribution_results_2 = distribution_results
    



    # Define Method for Distribution Analysis 3
    def distribution_analysis_3(self, data=None, column=None, feature=None, target=None, group=None, auto_set=True):

        '''
        Mehton calls the apply_outlier_analysis method from the DistributionAnalysis class.
        The method performs distribution analysis on a selected group in the selected column
        to determine its normality, skewness, and kurtosis.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        column: Column for subgroups
        feature: The columns to measure
        target: The column to subdivide data into groups
        group: The subgroup for the analysis
        auto_set: Determines if object variables will be set automatically

        Returns:
        --------
        distribution_results: dataset/list

        Outputs:
        --------
        figure
        histogram
        Q-Q plot
        box plot
        bar plot
        table
        '''

        # Set  Distribution Analysis Variables
        if auto_set == True:
            self.da_set_automatically()
            self.da.feature = 'days_past_due'
        else:
            self.da_set_manual(data=data, column=column, feature=feature, target=target, group=group)
        
        # Perform  Distribution Analysis
        distribution_results = self.da.apply_distribution_analysis()

        # Set Class Variables
        self.distribution_results_3 = distribution_results
    



    # Define Method for Distribution Analysis 4
    def distribution_analysis_4(self, data=None, column=None, feature=None, target=None, group=None, auto_set=True):

        '''
        Mehton calls the apply_outlier_analysis method from the DistributionAnalysis class.
        The method performs distribution analysis on a selected group in the selected column
        to determine its normality, skewness, and kurtosis.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        column: Column for subgroups
        feature: The columns to measure
        target: The column to subdivide data into groups
        group: The subgroup for the analysis
        auto_set: Determines if object variables will be set automatically

        Returns:
        --------
        distribution_results: dataset/list

        Outputs:
        --------
        figure
        histogram
        Q-Q plot
        box plot
        bar plot
        table
        '''

        # Set  Distribution Analysis Variables
        if auto_set == True:
            self.da_set_automatically()
            self.da.feature = 'days_past_due'
            self.da.group = 'second'
        else:
            self.da_set_manual(data=data, column=column, feature=feature, target=target, group=group)
        
        # Perform  Distribution Analysis
        distribution_results = self.da.apply_distribution_analysis()

        # Set Class Variables
        self.distribution_results_4 = distribution_results




    ############################## Transformation Analysis Methods ###############################



    # Define Method to Manually Set Transformation Analysis Variables
    def ta_set_manual(self, data=None, column=None, feature=None, group=None):
        
        '''
        Method manually sets class variables for the TransformationAnalysis object.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The columns to measure
        group: The subgroup for the analysis

        Returns:
        --------
        None

        Outputs:
        --------
        None
        '''
        
        # Set Object Variables
        self.ta.data = data
        self.ta.column = column
        self.ta.feature = feature
        self.ta.group = group
    



    # Define Method to Automatically Set Transformation Analysis Variables
    def ta_set_automatically(self):

        '''
        Method automatically sets class variables for the TransformationAnalysis object.
        
        Parameters:
        -----------
        self: The AnalysisMain object

        Returns:
        --------
        None

        Outputs:
        --------
        None
        '''

        # Set Object Variables
        self.ta.data = self.df_processed_data
        self.ta.column = 'dataset_name'
        self.ta.feature = 'hours_to_complete'
        self.ta.group = 'first'
    



    # Define Method for Transformation Analysis 1
    def transformation_analysis_1(self, data=None, column=None, feature=None, group=None, auto_set=True):

        '''
        Mehton calls the apply_transformation_analysis method from the TransformationAnalysis class.
        The method performs transformation analysis on the selected groups in the selected column
        and prints graphs and tables of the results.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        column: Column for subgroups
        feature: The columns to measure
        group: The subgroup for the analysis
        auto_set: Determines if object variables will be set automatically

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

        # Set Transformation Analysis Variables
        if auto_set == True:
            self.ta_set_automatically()
        else:
            self.ta_set_manual(data=data, column=column, feature=feature, group=group)
        
        # Perform Transformation Analysis
        transformation_results = self.ta.apply_transformation_analysis()

        # Set Class Variables
        self.transformation_results_1 = transformation_results
    



    # Define Method for Transformation Analysis 2
    def transformation_analysis_2(self, data=None, column=None, feature=None, group=None, auto_set=True):

        '''
        Mehton calls the apply_transformation_analysis method from the TransformationAnalysis class.
        The method performs transformation analysis on the selected groups in the selected column
        and prints graphs and tables of the results.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        column: Column for subgroups
        feature: The columns to measure
        group: The subgroup for the analysis
        auto_set: Determines if object variables will be set automatically

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

        # Set Transformation Analysis Variables
        if auto_set == True:
            self.ta_set_automatically()
            self.ta.group = 'second'
        else:
            self.ta_set_manual(data=data, column=column, feature=feature, group=group)
        
        # Perform Transformation Analysis
        transformation_results = self.ta.apply_transformation_analysis()

        # Set Class Variables
        self.transformation_results_2 = transformation_results
    



    # Define Method for Transformation Analysis 3
    def transformation_analysis_3(self, data=None, column=None, feature=None, group=None, auto_set=True):

        '''
        Mehton calls the apply_transformation_analysis method from the TransformationAnalysis class.
        The method performs transformation analysis on the selected groups in the selected column
        and prints graphs and tables of the results.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        column: Column for subgroups
        feature: The columns to measure
        group: The subgroup for the analysis
        auto_set: Determines if object variables will be set automatically

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

        # Set Transformation Analysis Variables
        if auto_set == True:
            self.ta_set_automatically()
            self.ta.feature = 'days_past_due'
        else:
            self.ta_set_manual(data=data, column=column, feature=feature, group=group)
        
        # Perform Transformation Analysis
        transformation_results = self.ta.apply_transformation_analysis()

        # Set Class Variables
        self.transformation_results_3 = transformation_results
    



    # Define Method for Transformation Analysis 4
    def transformation_analysis_4(self, data=None, column=None, feature=None, group=None, auto_set=True):

        '''
        Mehton calls the apply_transformation_analysis method from the TransformationAnalysis class.
        The method performs transformation analysis on the selected groups in the selected column
        and prints graphs and tables of the results.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        column: Column for subgroups
        feature: The columns to measure
        group: The subgroup for the analysis
        auto_set: Determines if object variables will be set automatically

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

        # Set Transformation Analysis Variables
        if auto_set == True:
            self.ta_set_automatically()
            self.ta.feature = 'days_past_due'
            self.ta.group = 'second'
        else:
            self.ta_set_manual(data=data, column=column, feature=feature, group=group)
        
        # Perform Transformation Analysis
        transformation_results = self.ta.apply_transformation_analysis()

        # Set Class Variables
        self.transformation_results_4 = transformation_results



    
    ############################## T-Test Analysis Methods #######################################



    # Define Method to Manually Set T-Test Analysis Variables
    def tt_set_manual(self, data=None, column=None, feature=None, transform=None):
        
        '''
        Method manually sets class variables for the TTestAnalysis object.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The columns to measure
        transform: Determines if the data is transformed

        Returns:
        --------
        None

        Outputs:
        --------
        None
        '''
        
        # Set Object Variables
        self.tt.data = data
        self.tt.column = column
        self.tt.feature = feature
        self.tt.transform = transform
    



    # Define Method to Automatically Set T-Test Analysis Variables
    def tt_set_automatically(self):

        '''
        Method automatically sets class variables for the TTestAnalysis object.
        
        Parameters:
        -----------
        self: The AnalysisMain object

        Returns:
        --------
        None

        Outputs:
        --------
        None
        '''

        # Set Object Variables
        self.tt.data = self.df_processed_data
        self.tt.column = 'dataset_name'
        self.tt.feature = 'hours_to_complete'
        self.tt.transform = False
    



    # Define Method for T-Test Analysis 1
    def independent_ttest_analysis_1(self, data=None, column=None, feature=None, transform=None, auto_set=True):

        '''
        Mehton calls the independent_ttest_analysis method from the TTestAnalysis class.
        The method performs an independent samples t-test on the selected groups 
        in the selected column and prints graphs and tables of the results.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        column: Column for subgroups
        feature: The columns to measure
        transform: Determines if the data is transformed
        auto_set: Determines if object variables will be set automatically

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

        # Set T-Test Analysis Variables
        if auto_set == True:
            self.tt_set_automatically()
        else:
            self.tt_set_manual(data=data, column=column, feature=feature, transform=transform)
        
        # Perform T-Test Analysis
        t_test_results = self.tt.independent_ttest_analysis()

        # Set Class Variables
        self.t_test_results_1 = t_test_results
    



    # Define Method for T-Test Analysis 2
    def independent_ttest_analysis_2(self, data=None, column=None, feature=None, transform=None, auto_set=True):

        '''
        Mehton calls the independent_ttest_analysis method from the TTestAnalysis class.
        The method performs an independent samples t-test on the selected groups 
        in the selected column and prints graphs and tables of the results.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        column: Column for subgroups
        feature: The columns to measure
        transform: Determines if the data is transformed
        auto_set: Determines if object variables will be set automatically

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

        # Set T-Test Analysis Variables
        if auto_set == True:
            self.tt_set_automatically()
            self.tt.feature = 'days_past_due'
        else:
            self.tt_set_manual(data=data, column=column, feature=feature, transform=transform)
        
        # Perform T-Test Analysis
        t_test_results = self.tt.independent_ttest_analysis()

        # Set Class Variables
        self.t_test_results_2 = t_test_results
    



    # Define Method for T-Test Analysis 3
    def independent_ttest_analysis_3(self, data=None, column=None, feature=None, transform=None, auto_set=True):

        '''
        Mehton calls the independent_ttest_analysis method from the TTestAnalysis class.
        The method performs an independent samples t-test on the selected groups 
        in the selected column and prints graphs and tables of the results.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        column: Column for subgroups
        feature: The columns to measure
        transform: Determines if the data is transformed
        auto_set: Determines if object variables will be set automatically

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

        # Set T-Test Analysis Variables
        if auto_set == True:
            self.tt_set_automatically()
            self.tt.transform = True
        else:
            self.tt_set_manual(data=data, column=column, feature=feature, transform=transform)
        
        # Perform T-Test Analysis
        t_test_results = self.tt.independent_ttest_analysis()

        # Set Class Variables
        self.t_test_results_3 = t_test_results
    



    # Define Method for T-Test Analysis 4
    def independent_ttest_analysis_4(self, data=None, column=None, feature=None, transform=None, auto_set=True):

        '''
        Mehton calls the independent_ttest_analysis method from the TTestAnalysis class.
        The method performs an independent samples t-test on the selected groups 
        in the selected column and prints graphs and tables of the results.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        column: Column for subgroups
        feature: The columns to measure
        transform: Determines if the data is transformed
        auto_set: Determines if object variables will be set automatically

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

        # Set T-Test Analysis Variables
        if auto_set == True:
            self.tt_set_automatically()
            self.tt.feature = 'days_past_due'
            self.tt.transform = True
        else:
            self.tt_set_manual(data=data, column=column, feature=feature, transform=transform)
        
        # Perform T-Test Analysis
        t_test_results = self.tt.independent_ttest_analysis()

        # Set Class Variables
        self.t_test_results_4 = t_test_results




    ############################## U-Test Analysis Methods #######################################



    # Define Method to Manually Set U-Test Analysis Variables
    def ut_set_manual(self, data=None, column=None, feature=None):
        
        '''
        Method manually sets class variables for the UTestAnalysis object.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The columns to measure

        Returns:
        --------
        None

        Outputs:
        --------
        None
        '''
        
        # Set Object Variables
        self.ut.data = data
        self.ut.column = column
        self.ut.feature = feature
    



    # Define Method to Automatically Set U-Test Analysis Variables
    def ut_set_automatically(self):

        '''
        Method automatically sets class variables for the UTestAnalysis object.
        
        Parameters:
        -----------
        self: The AnalysisMain object

        Returns:
        --------
        None

        Outputs:
        --------
        None
        '''

        # Set Object Variables
        self.ut.data = self.df_processed_data
        self.ut.column = 'dataset_name'
        self.ut.feature = 'hours_to_complete'
    



    # Define Method for U-Test Analysis 1
    def mannwhitney_utest_analysis_1(self, data=None, column=None, feature=None, auto_set=True):

        '''
        Mehton calls the mannwhitney_utest_analysis method from the UTestAnalysis class.
        The method performs the Mann-Whitney U Test on the selected groups 
        in the selected column and prints graphs and tables of the results.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        column: Column for subgroups
        feature: The columns to measure
        auto_set: Determines if object variables will be set automatically

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

        # Set U-Test Analysis Variables
        if auto_set == True:
            self.ut_set_automatically()
        else:
            self.ut_set_manual(data=data, column=column, feature=feature)
        
        # Perform U-Test Analysis
        u_test_results = self.ut.mannwhitney_utest_analysis()

        # Set Class Variables
        self.u_test_results_1 = u_test_results
    



    # Define Method for U-Test Analysis 2
    def mannwhitney_utest_analysis_2(self, data=None, column=None, feature=None, auto_set=True):

        '''
        Mehton calls the mannwhitney_utest_analysis method from the UTestAnalysis class.
        The method performs the Mann-Whitney U Test on the selected groups 
        in the selected column and prints graphs and tables of the results.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        column: Column for subgroups
        feature: The columns to measure
        auto_set: Determines if object variables will be set automatically

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

        # Set U-Test Analysis Variables
        if auto_set == True:
            self.ut_set_automatically()
            self.ut.feature = 'days_past_due'
        else:
            self.ut_set_manual(data=data, column=column, feature=feature)
        
        # Perform U-Test Analysis
        u_test_results = self.ut.mannwhitney_utest_analysis()

        # Set Class Variables
        self.u_test_results_2 = u_test_results




    ############################## Parametric-Nonparametric Analysis Methods #####################




    # Define Method to Manually Set Parametric-Nonparametric Analysis Variables
    def pnp_set_manual(self, data=None, column=None, feature=None, target=None):
        
        '''
        Method manually sets class variables for the ParametricNonparametricAnalysis object.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        data: Dataset to analyze
        column: Column for subgroups
        feature: The columns to measure
        target: The column to subdivide data into groups

        Returns:
        --------
        None

        Outputs:
        --------
        None
        '''
        
        # Set Object Variables
        self.pnp.data = data
        self.pnp.column = column
        self.pnp.feature = feature
        self.pnp.target = target
    



    # Define Method to Automatically Set Parametric-Nonparametric Analysis Variables
    def pnp_set_automatically(self):

        '''
        Method automatically sets class variables for the ParametricNonparametricAnalysis object.
        
        Parameters:
        -----------
        self: The AnalysisMain object

        Returns:
        --------
        None

        Outputs:
        --------
        None
        '''

        # Set Object Variables
        self.pnp.data = self.df_processed_data
        self.pnp.column = 'dataset_name'
        self.pnp.feature = 'hours_to_complete'
        self.pnp.target = 'priority'
    



    # Define Method for Parametric-Nonparametric Analysis 1
    def parametric_nonparametric_analysis_1(self, data=None, column=None, feature=None, target=None, auto_set=True):

        '''
        Mehton calls the parametric_nonparametric_analysis method from the ParametricNonparametricAnalysis class.
        The method performs analysis of parametric and non-parametric test results.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        column: Column for subgroups
        feature: The columns to measure
        target: The column to subdivide data into groups
        auto_set: Determines if object variables will be set automatically

        Returns:
        --------
        parametric_nonparametric_results: dictionary

        Outputs:
        --------
        figure
        tables
        '''

        # Set Parametric-Nonparametric Analysis Variables
        if auto_set == True:
            self.pnp_set_automatically()
        else:
            self.pnp_set_manual(data=data, column=column, feature=feature, target=target)
        
        # Perform Parametric-Nonparametric Analysis
        parametric_nonparametric_results = self.pnp.parametric_nonparametric_analysis()

        # Set Class Variables
        self.parametric_nonparametric_results_1 = parametric_nonparametric_results
    



    # Define Method for Parametric-Nonparametric Analysis 2
    def parametric_nonparametric_analysis_2(self, data=None, column=None, feature=None, target=None, auto_set=True):

        '''
        Mehton calls the parametric_nonparametric_analysis method from the ParametricNonparametricAnalysis class.
        The method performs analysis of parametric and non-parametric test results.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        column: Column for subgroups
        feature: The columns to measure
        target: The column to subdivide data into groups
        auto_set: Determines if object variables will be set automatically

        Returns:
        --------
        parametric_nonparametric_results: dictionary

        Outputs:
        --------
        figure
        tables
        '''

        # Set Parametric-Nonparametric Analysis Variables
        if auto_set == True:
            self.pnp_set_automatically()
            self.pnp.feature = 'days_past_due'
        else:
            self.pnp_set_manual(data=data, column=column, feature=feature, target=target)
        
        # Perform Parametric-Nonparametric Analysis
        parametric_nonparametric_results = self.pnp.parametric_nonparametric_analysis()

        # Set Class Variables
        self.parametric_nonparametric_results_2 = parametric_nonparametric_results




    ############################## Chi-Square Analysis Methods ###################################




    # Define Method to Manually Set Chi-Square Analysis Variables
    def cs_set_manual(self, data=None, column=None, target=None, alpha=None):
        
        '''
        Method manually sets class variables for the ChiSquareAnalysis object.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        data: Dataset to analyze
        column: Column for subgroups
        target: The column to cross-match against
        alpha: The alpha for measuring the chi-squared test

        Returns:
        --------
        None

        Outputs:
        --------
        None
        '''
        
        # Set Object Variables
        self.cs.data = data
        self.cs.column = column
        self.cs.target = target
        self.cs.alpha = alpha
    



    # Define Method to Automatically Set Chi-Square Analysis Variables
    def cs_set_automatically(self):

        '''
        Method automatically sets class variables for the ChiSquareAnalysis object.
        
        Parameters:
        -----------
        self: The AnalysisMain object

        Returns:
        --------
        None

        Outputs:
        --------
        None
        '''

        # Set Object Variables
        self.cs.data = self.df_processed_data
        self.cs.column = 'dataset_name'
        self.cs.target = 'sla_label'
        self.cs.alpha = 0.05
    



    # Define Method for Chi-Square Analysis 1
    def chi_square_test_analysis_1(self, data=None, column=None, target=None, alpha=None, auto_set=True):

        '''
        Mehton calls the chi_square_test_analysis method from the ChiSquareAnalysis class.
        The method performs chi-squared analysis on the column and target
        values in the data. The method displays a heatmap and tables of the
        test results.
        
        Parameters:
        -----------
        self: The AnalysisMain object
        column: Column for subgroups
        target: The column to cross-match against
        alpha: The alpha for measuring the chi-squared test
        auto_set: Determines if object variables will be set automatically

         Returns:
        --------
        chi_squared_results: dictionary

        Outputs:
        --------
        figure
        heatmap
        tables
        '''

        # Set Chi-Square Analysis Variables
        if auto_set == True:
            self.cs_set_automatically()
        else:
            self.cs_set_manual(data=data, column=column, target=target, alpha=alpha)
        
        # Perform Chi-Square Analysis
        chi_squared_results = self.cs.chi_square_test_analysis()

        # Set Class Variables
        self.chi_squared_results_1 = chi_squared_results















# End of Page
