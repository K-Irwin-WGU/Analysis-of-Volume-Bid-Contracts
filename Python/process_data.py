'''process_data'''


# IMPORT PACKAGES

import numpy as np
import pandas as pd

import math

import seaborn as sns
import matplotlib.pyplot as plt

from prettytable import PrettyTable
import warnings

# Suppress Warnings
import warnings
warnings.filterwarnings('ignore')




# CREATE PROCESS DATA CLASS

class ProcessData():

    '''
    Class loads data from CSV and processes the data 
    in order to prepare it for analysis.

    Returns:
    --------
    df_combined_data: dataframe

    Outputs:
    --------
    figure
    tables
    '''


    # Define init Method
    def __init__(self, data_file_1=None, data_file_2=None, 
                 data_file_3=None, folder_1=None, folder_2=None, 
                 dataset_1=None, dataset_2=None, dataset_3=None, unit_test=False):
        
        '''
        Method initializes class variables.
        
        Parameters:
        -----------
        self: The ProcessData object
        data_file_1: The name of the first CSV file to be loaded
        data_file_2: The name of the second CSV file to be loaded
        data_file_3: The name of the CSV file to be saved
        folder_1: The name of the folder containing the CSV files to be loaded
        folder_2: The name of the folder containing the CSV files to be saved
        dataset_1: The name of the first dataset to be loaded
        dataset_2: The name of the second dataset to be loaded
        dataset_3: The name of the dataset to be saved
        unit_test: Determines if unit tests are being performed
        '''
        
        # Initialize Class Variables
        self._data_file_1 = data_file_1
        self._data_file_2 = data_file_2
        self._data_file_3 = data_file_3
        self._folder_1 = folder_1
        self._folder_2 = folder_2
        self._dataset_1 = dataset_1
        self._dataset_2 = dataset_2
        self._dataset_3 = dataset_3
        self._unit_test = unit_test
    

        

    ############################## Getter Methods ##############################



    '''
    Methods get values from each class variable.
    '''

    # Getter for Data File 1
    @property
    def data_file_1(self):  
        return self._data_file_1
    

    # Getter for Data File 2
    @property
    def data_file_2(self):  
        return self._data_file_2
    

    # Getter for Data File 3
    @property
    def data_file_3(self):  
        return self._data_file_3
    

    # Getter for Folder 1
    @property
    def folder_1(self):  
        return self._folder_1
    

    # Getter for Folder 2
    @property
    def folder_2(self):  
        return self._folder_2


    # Getter for Dataset 1
    @property
    def dataset_1(self):  
        return self._dataset_1
    

    # Getter for Dataset 2
    @property
    def dataset_2(self):  
        return self._dataset_2
    

    # Getter for Dataset 3
    @property
    def dataset_3(self):  
        return self._dataset_3
    

    # Getter for Unit Test
    @property
    def unit_test(self):  
        return self._unit_test
    
        


    ############################## Setter Methods ##############################



    '''
    Methods set values for each class variable.
    '''

    # Define Setter for Data File 1 Variable
    @data_file_1.setter
    def data_file_1(self, file_name):
        if isinstance(file_name, str) and len(file_name) > 0:
            self._data_file_1 = file_name
        else:
            raise ValueError('Data file must be a non-empty string.')
    



    # Define Setter for Data File 2 Variable
    @data_file_2.setter
    def data_file_2(self, file_name):
        if isinstance(file_name, str) and len(file_name) > 0:
            self._data_file_2 = file_name
        else:
            raise ValueError('Data file must be a non-empty string.')
    



    # Define Setter for Data File 3 Variable
    @data_file_3.setter
    def data_file_3(self, file_name):
        if isinstance(file_name, str) and len(file_name) > 0:
            self._data_file_3 = file_name
        else:
            raise ValueError('Data file must be a non-empty string.')
    



    # Define Setter for Folder 1 Variable
    @folder_1.setter
    def folder_1(self, folder_name):
        if isinstance(folder_name, str) and len(folder_name) > 0:
            self._folder_1 = folder_name
        else:
            raise ValueError('Folder must be a non-empty string.')
    



    # Define Setter for Folder 2 Variable
    @folder_2.setter
    def folder_2(self, folder_name):
        if isinstance(folder_name, str) and len(folder_name) > 0:
            self._folder_2 = folder_name
        else:
            raise ValueError('Folder must be a non-empty string.')
    



    # Define Setter for Dataset 1 Variable
    @dataset_1.setter
    def dataset_1(self, dataset_name):
        if isinstance(dataset_name, str) and len(dataset_name) > 0:
            self._dataset_1 = dataset_name
        else:
            raise ValueError('Dataset must be a non-empty string.')
    



    # Define Setter for Dataset 2 Variable
    @dataset_2.setter
    def dataset_2(self, dataset_name):
        if isinstance(dataset_name, str) and len(dataset_name) > 0:
            self._dataset_2 = dataset_name
        else:
            raise ValueError('Dataset must be a non-empty string.')
    



    # Define Setter for Dataset 3 Variable
    @dataset_3.setter
    def dataset_3(self, dataset_name):
        if isinstance(dataset_name, str) and len(dataset_name) > 0:
            self._dataset_3 = dataset_name
        else:
            raise ValueError('Dataset must be a non-empty string.')
    



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
        self: The ProcessData object
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
        self: The ProcessData object
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




    ############################## Table Methods ###############################



    # Define Method to Plot Table Title
    def plot_table_title(self, title):

        '''
        Method creates an empty figure with a title.
        
        Parameters:
        -----------
        self: The ProcessData object
        title: The title of the plot

        Outputs:
        --------
        figure
        table
        '''

        # Create Subplots
        fig, ax = plt.subplots(1, 1, figsize=(10, .1))

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
        self: The ProcessData object
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
        self: The ProcessData object
        data: The dataset to plot
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
            fig, ax1 = plt.subplots(rows, columns, figsize=(10, 1))
        else:
            fig, axs = plt.subplots(rows, columns, figsize=(11, 1))

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



    # Define Method for Column Information
    def column_info(self, data, orig_columns):

        '''
        Method creates a dataframe containing the information about
        the columns in the dataset
        
        Parameters:
        -----------
        self: The ProcessData object
        dataframe: The dataframe for the column analysis
        orig_columns: List of columns in the original data

        Returns:
        --------
        df_column_info: dataframe
        '''


        # Create Columns List
        columns = list(data.columns)
        
        # Create List Variables
        column_types = []
        data_types = []
        null_values = []
        non_null_values = []
        unique_values = []

        # Iterate Through Columns
        for column in columns:
            if column in orig_columns:
                column_types. append('original feature')
            else:
                column_types.append('engineered feature')
            data_types.append(f'{data[column].dtypes}')
            null_values.append(data[column].isnull().sum())
            non_null_values.append(data[column].count())
            unique_values.append(data[column].nunique())

        # Create Dataframe for Column Information
        df_column_info = pd.DataFrame({
            'Column': columns,
            'Column Type': column_types,
            'Data Type': data_types,
            'Null Values': null_values,
            'Non-Null Values': non_null_values,
            'Unique Values': unique_values
        })

        # Create Data Type Dictionary
        data_type_dict = {
            'object': 'string',
            'int64': 'integer',
            'float64': 'float',
            'datetime64[ns]': 'datetime',
            'bool': 'boolean'
        }

        # Map Data Types
        df_column_info['Data Type'] = df_column_info['Data Type'].map(data_type_dict)

        # Return Dataframe
        return df_column_info




    ############################## Print Methods ###############################



    # Define Method to Print Loaded Data
    def print_loaded_data(self, data_1, data_2, dataset_1, dataset_2, orig_columns):

        '''
        Method prints a table of column information for each dataset loaded.
        
        Parameters:
        -----------
        self: The ProcessData object
        data_1: The first dataset to print column information about
        data_2: The second dataset to print column information about
        dataset_1: The name of the first dataset
        dataset_2: The name of the second dataset
        orig_columns: List of columns in the original data

        Outputs:
        --------
        figure
        table 1
        table 2
        '''

        # Get Column Information
        column_info_1 = self.column_info(data_1, orig_columns)
        column_info_2 = self.column_info(data_2, orig_columns)

        # Geta Data Shapes
        shape_1 = data_1.shape
        shape_2 = data_2.shape

        # Create Title Variables
        dataset_title_1 = f'Dataset 1 - {dataset_1} - {shape_1[0]} Rows, {shape_1[1]} Columns'
        dataset_title_2 = f'\nDataset 2 - {dataset_2} - {shape_2[0]} Rows, {shape_2[1]} Columns'

        # Create Section Header
        section_header = f'\nOverview of Loaded Data'

        # Plot Section Header
        self.plot_table_title(section_header)

        # Plot Dataset Tables
        dataset_col_widths = [0.2, 0.2, 0.15, 0.15, 0.15, 0.15]
        self.plot_table_data([column_info_1], [dataset_title_1], dataset_col_widths, 'left')
        self.plot_table_data([column_info_2], [dataset_title_2], dataset_col_widths, 'left')
    



    # Define Method to Print Processed Data
    def print_processes_data(self, data, dataset, orig_columns):

        '''
        Method prints a table of column information for each dataset loaded.
        
        Parameters:
        -----------
        self: The ProcessData object
        data: The dataset to print column information about
        dataset: The name of the dataset
        orig_columns: List of columns in the original data

        Outputs:
        --------
        figure
        table
        '''

        # Get Column Information
        column_info = self.column_info(data, orig_columns)

        # Geta Data Shape
        shape = data.shape

        # Create Title Variable
        dataset_title = f'Dataset - {dataset} - {shape[0]} Rows, {shape[1]} Columns'

        # Create Section Header
        section_header = f'\nOverview of Processed Data'

        # Plot Section Header
        self.plot_table_title(section_header)

        # Plot Dataset Tables
        dataset_col_widths = [0.2, 0.2, 0.15, 0.15, 0.15, 0.15]
        self.plot_table_data([column_info], [dataset_title], dataset_col_widths, 'left')




    ############################## Load Data Methods ###########################



    # Define Method to Load Data File
    def load_data_file(self, file, folder, dataset_number, dataset_name):

        '''
        Method reads a CSV file and loads data into a dataframe.
        Method returns a dataframe and a list of column names.
        
        Parameters:
        -----------
        self: The ProcessData object
        file: The name of the CSV file
        folder: The name of the folder containing the CSV file
        dataset_number: The number of the dataset being loaded
        dataset_name: The name of the dataset being loaded

        Returns:
        --------
        df_data: dataframe
        orig_columns: list
        '''

        # Load Data
        df_data = pd.read_csv(f'../{folder}/{file}', low_memory=False)

        # Create Columns List
        orig_columns = list(df_data.columns)

        # Create Dataset Identification Columns
        df_data['dataset_number'] = dataset_number
        df_data['dataset_name'] = dataset_name

        # Return Dataframe
        return df_data, orig_columns
    



    # Define Method to Load All Data
    def load_data(self):

        '''
        Method gets data from CSV files and combines data into a dataframe.
        The method returns a dataframe and a list of the original column names.
        
        Parameters:
        -----------
        self: The ProcessData object
        
        Class Variables:
        -----------
        data_file_1: The name of the first CSV file
        data_file_2: The name of the second CSV file
        folder: The name of the folder containing the CSV files
        dataset_1: The name of the first dataset
        dataset_2: The name of the second dataset
        unit_test: Determines if unit tests are being performed

        Returns:
        --------
        df_combined_data: dataframe
        orig_columns: list
        '''

        # Set Data Variables
        data_file_1 = self.data_file_1
        data_file_2 = self.data_file_2
        folder = self.folder_1
        dataset_1 = self.dataset_1
        dataset_2 = self.dataset_2
        unit_test = self.unit_test

        # Load Data File 1 Data
        df_data_1, orig_columns = self.load_data_file(data_file_1, folder, 1, dataset_1)

        # Load Data File 2 Data
        df_data_2, orig_columns = self.load_data_file(data_file_2, folder, 2, dataset_2)

        # Plot Dataset Tables
        if unit_test == False:
            self.print_loaded_data(df_data_1, df_data_2, dataset_1, dataset_2, orig_columns)

        # Concatenate Data 1 and Data 2 Dataframes
        df_combined_data = pd.concat([df_data_1, df_data_2], axis=0)

        # Return Data
        return df_combined_data, orig_columns




    ############################## Statistic Methods ###########################



    # Define Method to Convert to Datetime
    def convert_datetime(self, df_data):

        '''
        Method converts timestamp columns into datetime columns
        and adds them to the dataframe.
        
        Parameters:
        -----------
        self: The ProcessData object
        dataframe: The dataframe for processing

        Returns:
        --------
        df_data: dataframe
        '''

        # Convert Timestamps to Datetime
        df_data['dateSubmitted'] = pd.to_datetime(df_data['date_submitted'], format='mixed')
        df_data['dateCompleted'] = pd.to_datetime(df_data['date_completed'], format='mixed')
        df_data['dueDate'] = pd.to_datetime(df_data['due_date'], format='mixed')

        # Return Dataframe
        return df_data
    



    # Define Method to Add Hours to Complete
    def add_hours_to_complete(self, df_data):

        '''
        Method calculates the hours to complete for each data entry
        and adds it to the dataframe.
        
        Parameters:
        -----------
        self: The ProcessData object
        dataframe: The dataframe for processing

        Returns:
        --------
        df_data: dataframe
        '''

        # Create Time to Complete Column
        df_data['hours_to_complete'] = (df_data['dateCompleted'] - df_data['dateSubmitted']).dt.total_seconds()
        df_data['hours_to_complete'] = df_data['hours_to_complete'] / 60 # Returns Minutes
        df_data['hours_to_complete'] = df_data['hours_to_complete'] / 60 # Returns Hours
        #df_data['hours_to_complete'] = df_data['hours_to_complete'] / 24 # Returns Days
        df_data['hours_to_complete'] = df_data['hours_to_complete'].round(2)

        # Return Dataframe
        return df_data
    



    # Define Method to Add Days Past Due
    def add_days_past_due(self, df_data):

        '''
        Method calculates the days past due for each data entry
        and adds it to the dataframe.
        
        Parameters:
        -----------
        self: The ProcessData object
        dataframe: The dataframe for processing

        Returns:
        --------
        df_data: dataframe
        '''

        # Create Days Past Due Column
        df_data['days_past_due'] = (df_data['dateCompleted'] - df_data['dueDate']).dt.total_seconds()
        df_data['days_past_due'] = df_data['days_past_due'] / 60 # Returns Minutes
        df_data['days_past_due'] = df_data['days_past_due'] / 60 # Returns Hours
        df_data['days_past_due'] = df_data['days_past_due'] / 24 # Returns Days
        df_data['days_past_due'] = df_data['days_past_due'].round(2)

        # Return Dataframe
        return df_data
    



    # Define Method to Add Service Level Columns
    def add_service_levels(self, df_data):

        '''
        Method calculates the SLA label for each data entry
        and adds it to the dataframe.
        
        Parameters:
        -----------
        self: The ProcessData object
        dataframe: The dataframe for processing

        Returns:
        --------
        df_data: dataframe
        '''

        # Add Completed On Time Column
        df_data['completed_on_time'] = df_data.apply(lambda x: x['dateCompleted'] < x['dueDate'], axis=1)

        # Create SLA Group Column
        df_data['sla_group'] = df_data['completed_on_time'].map({True: 1, False: 0})

        # Create SLA Label Column
        df_data['sla_label'] = df_data['completed_on_time'].map({True: 'Met', False: 'Not Met'})

        # Return Dataframe
        return df_data




    ############################## Main Method #################################



    # Define Method to Process Data
    def process_data(self, data=None, columns=[]):

        '''
        Method loade data from CSV, or uses data provided, and
        processes the data in order to prepare it for analysis.
        
        Parameters:
        -----------
        self: The ProcessData object
        data: The dataset to be processed
        orig_columns: List of columns in the original data

        Class Variables:
        -----------
        data_file: The name of the CSV file
        folder: The name of the folder containing the CSV files
        dataset_3: The name of the dataset
        unit_test: Determines if unit tests are being performed

        Returns:
        --------
        df_combined_data: dataframe

        Outputs:
        --------
        figure
        tables
        '''

        # Set Dataframe Variable
        if isinstance(data, pd.DataFrame):
            df_combined_data = data
            orig_columns = columns
        else:
            df_combined_data, orig_columns = self.load_data()
        
        # Set Data File Variables
        data_file = self.data_file_3
        folder = self.folder_2
        dataset_3 = self.dataset_3

        # Set Unit Test Variable
        unit_test = self.unit_test
        
        # Convert Timestamps to Datetime
        df_combined_data = self.convert_datetime(df_combined_data)

        # Add Hours to Complete Column
        df_combined_data = self.add_hours_to_complete(df_combined_data)

        # Define Method to Add Days Past Due Column
        df_combined_data = self.add_days_past_due(df_combined_data)

        # Add Service Level Columns
        df_combined_data = self.add_service_levels(df_combined_data)

        # Print Column Information
        if unit_test == False:
            self.print_processes_data(df_combined_data, dataset_3, orig_columns)

        # Sort by Work Order ID, Date Submitted, and Date Complete
        df_combined_data = df_combined_data.sort_values(by=['work_order_id','date_completed','date_submitted'],
                                                  ascending = (True, True, True))
        
        # Reset Index
        df_combined_data = df_combined_data.reset_index(drop=True)

        # Save Dataset to CSV
        df_combined_data.to_csv(f'../{folder}/{data_file}', encoding='utf-8', index=False)

        # Return Processed Data
        return df_combined_data
















# End of Page

