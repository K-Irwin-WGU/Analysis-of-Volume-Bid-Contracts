'''test_analysis'''


# IMPORT PACKAGES

import unittest

import numpy as np
import pandas as pd

from greetings import greeting
from process_data import ProcessData
from visual_analysis import VisualAnalysis
from outlier_analysis import OutlierAnalysis
from distribution_analysis import DistributionAnalysis
from transformation_analysis import TransformationAnalysis
from t_test_analysis import TTestAnalysis
from u_test_analysis import UTestAnalysis
from parametric_nonparametric_analysis import ParametricNonparametricAnalysis
from chi_square_analysis import ChiSquareAnalysis

# Suppress Warnings
import warnings
warnings.filterwarnings('ignore')




'''
This file contains the unit tests for the data processing
and eight analysis classes of the project application.

Contents
--------
1. Test Greeting
2. Test Process Data
3. Test Visual Analysis
4. Test Outlier Analysis
5. Test Distribution Analysis
6. Test Transformation Analysis
7. Test T-Test Analysis
8. Test U Test Analysis
9. Test Parametric and Non-Parametric Analysis
10. Test Chi-Square Analysis
'''




# Define Class to Test Greeting
class TestGreeting(unittest.TestCase):

    '''
    This class performs unit tests on the greetings method for the application.
    '''

    def test_greeting(self):
        self.assertEqual(greeting(), 'Hello World!')




# Define Class to Test Process Data Class
class TestProcessData(unittest.TestCase):

    '''
    This class performs unit tests on the ProcessData class of the application.
    '''

    # Define Method to Test Load Data
    def test_load_data(self):

        '''
        Method tests load_data method for correct data types returned.
        '''
        
        # Create Process Data Object
        pd_test = ProcessData()

        # Set Class Variables
        pd_test.data_file_1 = 'Work_Completed_Set1.csv'
        pd_test.data_file_2 = 'Work_Completed_Set2.csv'
        pd_test.folder_1 = 'Data'
        pd_test.dataset_1 = 'Historic Data'
        pd_test.dataset_2 = 'Current Data'
        pd_test.unit_test = True

        # Run Load Data Method
        df_loaded_data, loaded_data_columns = pd_test.load_data()

        # Test Correct Data Types
        assert type(df_loaded_data) == pd.DataFrame
        assert type(loaded_data_columns) == list




    # Define Method to Test Process Data
    def test_process_data(self):
        
        '''
        Method tests process_data method for correct data type returned.
        '''

         # Create ProcessData Object
        pd_test = ProcessData()

        # Set Class Variables
        pd_test.data_file_1 = 'Work_Completed_Set1.csv'
        pd_test.data_file_2 = 'Work_Completed_Set2.csv'
        pd_test.data_file_3 = 'Work_Completed_Data.csv'
        pd_test.folder_1 = 'Data'
        pd_test.folder_2 = 'Processed-Data'
        pd_test.dataset_1 = 'Historic Data'
        pd_test.dataset_2 = 'Current Data'
        pd_test.dataset_3 = 'Combined Data'
        pd_test.unit_test = True

        # Run Process Data Method
        df_processed_data = pd_test.process_data()

        # Test Correct Data Type
        assert type(df_processed_data) == pd.DataFrame




# Define Class to Test Visual Analysis Class
class TestVisualAnalysis(unittest.TestCase):

    '''
    This class performs unit tests on the VisualAnalysis class of the application.
    '''

    # Define Method to Test Histogram Plot Stats
    def test_histogram_plot_stats(self):

        # Load Data
        df_Completed_Work = pd.read_csv('../Processed-Data/Work_Completed_Data.csv', low_memory=False)

        # Create Visual Analysis Object
        va_test = VisualAnalysis()

        # Set Class Variables
        va_test.data = df_Completed_Work
        va_test.column = 'dataset_name'
        features = ['hours_to_complete', 'days_past_due']
        va_test.features = features

        # Run Histogram Plot Stats
        df_stats = va_test.histogram_plot_stats(df_Completed_Work, 'dataset_name', features)

        # Test Correct Data Type
        assert type(df_stats) == pd.DataFrame
    



    # Define Method to Test Box Plot Stats
    def test_box_plot_stats(self):

        # Load Data
        df_Completed_Work = pd.read_csv('../Processed-Data/Work_Completed_Data.csv', low_memory=False)

        # Create Visual Analysis Object
        va_test = VisualAnalysis()

        # Set Class Variables
        va_test.data = df_Completed_Work
        va_test.column = 'dataset_name'
        features = ['hours_to_complete', 'days_past_due']
        va_test.features = features

        # Run Box Plot Stats
        df_stats = va_test.box_plot_stats(df_Completed_Work, 'dataset_name', features)

        # Test Correct Data Type
        assert type(df_stats) == pd.DataFrame
    



    # Define Method to Test Count Plot Stats
    def test_cout_plot_stats(self):

        # Load Data
        df_Completed_Work = pd.read_csv('../Processed-Data/Work_Completed_Data.csv', low_memory=False)

        # Create Visual Analysis Object
        va_test = VisualAnalysis()

        # Set Class Variables
        va_test.data = df_Completed_Work
        va_test.column = 'dataset_name'
        va_test.target = 'priority'
        va_test.target_slice = 2
        va_test.unit_test = True

        # Run Count Plot Stats
        df_stats = va_test.cout_plot_stats(df_Completed_Work, 'dataset_name', 'priority', 2)

        # Test Correct Data Type
        assert type(df_stats) == pd.DataFrame
    



    # Define Method to Test Violin Plot Stats
    def test_violin_plot_stats(self):

        # Load Data
        df_Completed_Work = pd.read_csv('../Processed-Data/Work_Completed_Data.csv', low_memory=False)

        # Create Visual Analysis Object
        va_test = VisualAnalysis()

        # Set Class Variables
        va_test.data = df_Completed_Work
        va_test.column = 'dataset_name'
        va_test.feature_1 = 'hours_to_complete'
        va_test.target = 'priority'
        va_test.unit_test = True

        # Run Violin Plot Stats
        df_stats = va_test.violin_plot_stats(df_Completed_Work, 'dataset_name', 'hours_to_complete', 'priority')

        # Test Correct Data Type
        assert type(df_stats) == pd.DataFrame
    



    # Define Method to Test Statistical Summary
    def test_statistic_summary(self):

        # Load Data
        df_Completed_Work = pd.read_csv('../Processed-Data/Work_Completed_Data.csv', low_memory=False)

        # Create Visual Analysis Object
        va_test = VisualAnalysis()

        # Set Class Variables
        va_test.data = df_Completed_Work
        va_test.column = 'dataset_name'
        va_test.feature_1 = 'hours_to_complete'
        va_test.target = 'sla_label'
        va_test.unit_test = True

        # Run Statistical Summary
        summary_stats = va_test.statistic_summary()

        # Test Correct Data Type
        assert type(summary_stats) == list
        assert type(summary_stats[0]) == pd.DataFrame
        assert type(summary_stats[1]) == pd.DataFrame
        assert type(summary_stats[2]) == pd.DataFrame




# Define Class to Test Outlier Analysis Class
class TestOutlierAnalysis(unittest.TestCase):

    '''
    This class performs unit tests on the OutlierAnalysis class of the application.
    '''

    # Define Method to Test Outlier Analysis
    def test_outlier_analysis(self):

        # Load Data
        df_Completed_Work = pd.read_csv('../Processed-Data/Work_Completed_Data.csv', low_memory=False)

        # Create Outlier Analysis Object
        oa_test = OutlierAnalysis()

        # Set Class Variables
        oa_test.data = df_Completed_Work
        oa_test.column = 'dataset_name'
        oa_test.feature = 'hours_to_complete'
        oa_test.group = 'first'
        oa_test.unit_test = True

        # Perform Outlier Analysis
        outlier_results = oa_test.apply_outlier_analysis()

        # Test Correct Data Type
        assert type(outlier_results) == pd.Series




# Define Class to Test Distribution Analysis Class
class TestDistributionAnalysis(unittest.TestCase):

    '''
    This class performs unit tests on the DistributionAnalysis class of the application.
    '''

    # Define Method to Test Distribution Analysis
    def test_distribution_analysis(self):

        # Load Data
        df_Completed_Work = pd.read_csv('../Processed-Data/Work_Completed_Data.csv', low_memory=False)

        # Create Distribution Analysis Object
        da_test = DistributionAnalysis()

        # Set Class Variables
        da_test.data = df_Completed_Work
        da_test.column = 'dataset_name'
        da_test.feature = 'hours_to_complete'
        da_test.target = 'priority'
        da_test.group = 'first'
        da_test.unit_test = True
        
        # Perform Distribution Analysis
        distribution_results = da_test.apply_distribution_analysis()

        # Test Correct Data Type
        assert type(distribution_results) == list




# Define Class to Test Transformation Analysis Class
class TestTransformationAnalysis(unittest.TestCase):

    '''
    This class performs unit tests on the TransformationAnalysis class of the application.
    '''

    # Define Method to Test Transformation Analysis
    def test_transformation_analysis(self):

        # Load Data
        df_Completed_Work = pd.read_csv('../Processed-Data/Work_Completed_Data.csv', low_memory=False)

        # Create Transformation Analysis Object
        ta_test = TransformationAnalysis()

        # Set Class Variables
        ta_test.data = df_Completed_Work
        ta_test.column = 'dataset_name'
        ta_test.feature = 'hours_to_complete'
        ta_test.group = 'first'
        ta_test.unit_test = True
        
        # Perform Transformation Analysis
        transformation_results = ta_test.apply_transformation_analysis()

        # Test Correct Data Type
        assert type(transformation_results) == list




# Define Class to Test T-Test Analysis Class
class TestTTestAnalysis(unittest.TestCase):

    '''
    This class performs unit tests on the TTestAnalysis class of the application.
    '''

    # Define Method to Test T-Test Analysis
    def test_ttest_analysis(self):

        # Load Data
        df_Completed_Work = pd.read_csv('../Processed-Data/Work_Completed_Data.csv', low_memory=False)

        # Create T-Test Analysis Object
        tt_test = TTestAnalysis()

        # Set Class Variables
        tt_test.data = df_Completed_Work
        tt_test.column = 'dataset_name'
        tt_test.feature = 'hours_to_complete'
        tt_test.transform = False
        tt_test.unit_test = True
        
        # Perform Independent T-test Analysis
        t_test_results = tt_test.independent_ttest_analysis()

        # Test Correct Data Type
        assert type(t_test_results) == dict




# Define Class to Test U Test Analysis Class
class TestUTestAnalysis(unittest.TestCase):

    '''
    This class performs unit tests on the UTestAnalysis class of the application.
    '''

    # Define Method to Test U Test Analysis
    def test_utest_analysis(self):

        # Load Data
        df_Completed_Work = pd.read_csv('../Processed-Data/Work_Completed_Data.csv', low_memory=False)

        # Create U Test Analysis Object
        ut_test = UTestAnalysis()

        # Set Class Variables
        ut_test.data = df_Completed_Work
        ut_test.column = 'dataset_name'
        ut_test.feature = 'hours_to_complete'
        ut_test.transform = False
        ut_test.unit_test = True
        
        # Perform U test Analysis
        u_test_results = ut_test.mannwhitney_utest_analysis()

        # Test Correct Data Type
        assert type(u_test_results) == dict




# Define Class to Test Parametric and Non-Parametric Analysis Class
class TestParametricNonparametricAnalysis(unittest.TestCase):

    '''
    This class performs unit tests on the ParametricNonparametricAnalysis class of the application.
    '''

    # Define Method to Test Parametric and Non-Parametric Analysis
    def test_parametric_nonparametric_analysis(self):

        # Load Data
        df_Completed_Work = pd.read_csv('../Processed-Data/Work_Completed_Data.csv', low_memory=False)

        # Create Parametric and Non-Parametric Analysis Object
        pnp_test = ParametricNonparametricAnalysis()

        # Set Class Variables
        pnp_test.data = df_Completed_Work
        pnp_test.column = 'dataset_name'
        pnp_test.feature = 'hours_to_complete'
        pnp_test.target = 'priority'
        pnp_test.unit_test = True
        
        # Perform Parametric and Non-Parametric Analysis
        parametric_nonparametric_results = pnp_test.parametric_nonparametric_analysis()

        # Test Correct Data Type
        assert type(parametric_nonparametric_results) == dict




# Define Class to Test Chi-Square Analysis Class
class TestChiSquareAnalysis(unittest.TestCase):

    '''
    This class performs unit tests on the ChiSquareAnalysis class of the application.
    '''

    # Define Method to Test Chi-Square Analysis
    def test_chi_square_analysis(self):

        # Load Data
        df_Completed_Work = pd.read_csv('../Processed-Data/Work_Completed_Data.csv', low_memory=False)

        # Create Chi-Square Analysis Object
        cs_test = ChiSquareAnalysis()

        # Set Class Variables
        cs_test.data = df_Completed_Work
        cs_test.column = 'dataset_name'
        cs_test.target = 'sla_label'
        cs_test.alpha = 0.05
        cs_test.unit_test = True
        
        # Perform Chi-Square Analysis
        chi_squared_results = cs_test.chi_square_test_analysis()

        # Test Correct Data Type
        assert type(chi_squared_results) == dict
       









if __name__ == "__main__":

    # Perform Unit Tests
    unittest.main(verbosity=2)
















# End of Page
