#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 15:53:05 2023


"""


# Python for Data Analysis - Basics



# Import Python APIs

# Import Pandas API - used for data processing
import pandas as pd

# Import OS package - used for reading and writing files
import os

# Import Numpy API - used for Array manipulation
import numpy as np

# Import Matplotlib API - used to create plots
import matplotlib.pyplot as plt

# Import Useful Functions script
import sys
sys.path.append('/home/juliana/Документи/Experian - ML Study/Code')

from Experian_Data_Functions import *




# Folder Directories
data_path = r"/home/juliana/Документи/Experian - ML Study/Data"

reports_path = r"/home/juliana/Документи/Experian - ML Study/Reports/Python Basics"



# Data File names
bank_case_study_file = r"BankCaseStudyData.csv"




# Example 1: Read Data
data = pd.read_csv(os.path.join(data_path, bank_case_study_file),
                   encoding = 'utf8',
                   low_memory = False)


# Arguments of read_csv function used in the example:
# encoding - 'utf8' - determines ASCII file encoding
# low_memory - forces python to determine the data type of each column in dataframe



# Example 2: Read Data by setting up a specific data type for each column
attributes = {
    'Account_Number' : 'object',
    'Gross_Annual_Income' : 'float32',
    'Home_Telephone_Number' : 'category',
    'Age_of_Applicant' : 'int32',
    'GB_Flag' : 'category'}


data = pd.read_csv(os.path.join(data_path, bank_case_study_file),
                   encoding = 'utf8',
                   low_memory = False,
                   usecols = attributes.keys(),
                   dtype = attributes)

# Arguments of read_csv function used in the example:
# encoding - 'utf8' - determines ASCII file encoding
# low_memory - forces python to determine the data type of each column in dataframe
# usecols - specifies which columns should be read and stored in data from the csv file
# dtype - specifies data type of each column in data




# In Python we often cannot see the entire data set like in WPS

# Some useful tricks on getting to know your data

# Export Data columns
Columns_List(data, reports_path, r"Bank Case Study - Columns.csv")



# Export Subsample
View_Subsample(data, 100, reports_path, r" Bank Case Study - Subsample.csv")



# Change Good Bad Flag values to match the Python Toolbox

# Create a Dictionary for the new GB Flag values
# Dictionary structure: 'old_value' : 'new_value'
gb_flag_values = {
    'Good' : 'G',
    'NTU' : 'X',
    'Rejects' : 'X',
    'Bad' : 'B',
    'Indeterminate' : 'I'}


# Create new GB Flag
data['New_GB_Flag'] = Encode_GB_Flag(data, 'GB_Flag', gb_flag_values)

print(data['New_GB_Flag'].value_counts())





# Basic Data Processing functions from WPS


# FREQUENCY

# Display Frequency in Spyder
print(data['Marital_Status'].value_counts())


# Export Frequency
Frequency(data,
          'Marital_Status',
          reports_path,
          r"BCS - Marital Status.csv")



# Export Banded Frequency
Banded_Frequency(data,
                 'Bureau_Score',
                 [-np.inf, 680, 760, 840, 920, np.inf],
                 reports_path,
                 r"BCS - Bureau Score - Banded Frequency.csv")



# CROSSTAB

# Regular Crosstab
crosstab = pd.crosstab(data['Final_Decision'],
                       data['GB_Flag'],
                       margins = 'Total',
                       dropna = False)


# Export Regular Crosstab
Crosstab(data,
         'Final_Decision',
         'GB_Flag',
         reports_path,
         r"BCS - Crosstab.csv")



# Export Banded Crosstab
Banded_Crosstab(data,
                'Bureau_Score',
                'Application_Score',
                [-np.inf, 680, 760, 840, 920, np.inf],
                reports_path,
                r"BCS - Banded Crosstab.csv")




# FILTER DATA

# Based on column values
new_data_1 = data[(data['GB_Flag'] == 'Good') | 
                (data['GB_Flag'] == 'Bad')].copy()

# This syntax checks if the value of the GB_Flag column is equal to
# Good or Bad. The '|' symbol represents logical OR
# This will remove all rows where GB_Flag is not equal to Good or Bad


# Alternative syntax
new_data_2 = data[(data['GB_Flag'].isin(['Good', 'Bad']))].copy()

# This syntax checks the values of GB_Flag and compares them to a list
# of values we wish to keep. In this example ['Good', 'Bad'].
# This will remove all rows where GB_Flag is not equal to Good or Bad


# Greater than a certain value
new_data_3 = data[data['Age_of_Applicant'] >= 25].copy()

# This will remove all rows where Age of Applicant is less than 25 years


# Remove values based on condition
new_data_4 = data[(data['GB_Flag'] != 'NTU')].copy()

# This code checks if the value of GB Flag is NOT equal to NTU
# This removes rows where GB Flag is equal to NTU




# MANIPULATE DATAFRAMES


# Keep only certain columns
list_of_columns = ['Account_Number',
                  'Account_Type',
                  'Final_Decision']


new_data_5 = data[list_of_columns].copy()
# This keeps only columns given in the list_of_columns list




# Create a new column
data['Target'] = np.where(data['GB_Flag'] == 'Good', 1, 0)

# np.where is a speed up version of if-else statement
# The structure is: np.where(condition, outcome if True, outcome if False)

# This creates a new column Target that has a value of 1 if GB_Flag is Good
# and 0 if GB_Flag is of any other value



# Overwrite existing column
print(data['Existing_Customer_Flag'].value_counts())

data['Existing_Customer_Flag'] = ''

print(data['Existing_Customer_Flag'].value_counts())

# This statement will overwrite all values of the Existing_Customer_Flag column
# Be careful with this syntax



# Create a dataframe

# DataFrame structure: 'column_name' : [values]

dataframe_1 = pd.DataFrame(data = {
    'UniqueID' : [1,2,3,4],
    'Column 2' : [5,6,7,8]})


dataframe_2 = pd.DataFrame(data = {
    'UniqueID' : [1,2,3,4],
    'Column 3' : ['A','B','C','D']})


# MERGE
dataframe_3 = pd.merge(left = dataframe_1,
                       right = dataframe_2,
                       how = 'inner',
                       on = 'UniqueID')

# Merge two dataframes by keeping all columns from the left dataframe
# and mapping the right dataframe accordingly.
# The 'inner' option will only match the values from the right dataframe
# with UniqueID values present in both dataframes

# if you use pd.Merge, left and right dataframes CAN be of different shapes



# CONCAT

# Concatenate dataframes by row
dataframe_4 = pd.concat([dataframe_1, dataframe_2],
                        join = 'outer',
                        axis = 0,
                        ignore_index = True)


# Concatenate dataframes by column
dataframe_5 = pd.concat([dataframe_1, dataframe_2],
                        join = 'outer',
                        axis = 1)

# The axis argument determines if dataframes are concatenated by
# row or column index. 
# Axis = 0 - concatenate by row index
# Axis = 1 - concatenate by column index



# PIVOT TABLE
pivot_table = pd.pivot_table(data,
                             values = 'Gross_Annual_Income',
                             index = 'Age_of_Applicant',
                             columns = 'GB_Flag',
                             aggfunc = np.mean,
                             fill_value=(0))

# Export a Pivot table
Pivot_Table(dataframe = data,
            values_col = 'Gross_Annual_Income',
            index_col = 'Marital_Status',
            columns_col = 'GB_Flag',
            function = np.sum,
            path = reports_path,
            file_name = r"BCS - Pivot Table.csv")



# Statistics

# Descriptive Statistics
print(data['Gross_Annual_Income'].describe())


# Mean
print(data['Gross_Annual_Income'].mean())


# Sum
print(data['Gross_Annual_Income'].sum())



# CHARTS

# Create Pie Chart
Pie_Chart(data,
          'Marital_Status',
          reports_path,
          r"BCS - Pie Chart.png")


# Create Bar Chart
Bar_Chart(data,
          'GB_Flag',
          'Good Bad Flag',
          reports_path,
          r"BCS - Bar Chart.png")


# Scatter Plot
Scatter_Plot(data,
             'Age_of_Applicant',
             'Bureau_Score',
             'Bureau Score against Age of Applicant',
             reports_path,
             r"BCS - Scatter Plot.png")


# Density Plot
Density_Plot(data,
             'Age_of_Applicant',
             'Age of Applicant - Distribution',
             reports_path,
             r"BCS - Density Plot.png")

# Good Luck!