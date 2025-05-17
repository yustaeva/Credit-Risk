#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 15:54:33 2023


"""



# Python for Data Analysis - Useful Functions



# Import Python APIs

# Import Pandas API - used for data processing
import pandas as pd

# Import OS package - used for reading and writing files
import os

# Import Numpy API - used for Array manipulation
import numpy as np

# Import Matplotlib API - used to create plots
import matplotlib.pyplot as plt




# In Python we often cannot see the entire data set like in WPS

# Some useful tricks on getting to know your data

# Export a list of all file columns in a csv file:
def Columns_List(dataframe, path, file_name):
    
    # Create a list of dataframe columns
    columns = pd.DataFrame(dataframe.columns)
    
    # Export list of dataframe columns
    columns.to_csv(os.path.join(path, file_name))


# Export a Subsample of the data file:
def View_Subsample(dataframe, subsample_size, path, file_name):
    
    # Create a Subsample of the data
    
    # The iloc (index location) method is used to locate elements of a dataframe
    subsample = dataframe.iloc[: subsample_size]
    
    # Export Subsample of the data
    subsample.to_csv(os.path.join(path, file_name))


# Define a function to create new GB Flag
def Encode_GB_Flag(dataframe, old_gb_flag, gb_flag_dict):
    
    # Map Dictionary of old and new GB Flag values to data
    
    # The map method is used to apply a dictionary of new values to a dataframe column
    return dataframe[old_gb_flag].map(gb_flag_dict)



# Save Frequency to a csv file
def Frequency(dataframe, column_name, path, file_name):
    
    # Calculate Frequency of a variable
    frequency = pd.DataFrame(dataframe[column_name].value_counts())
    
    # Export Frequency
    frequency.to_csv(os.path.join(path, file_name))



# Frequency of a Numerical variable (with bands)
def Banded_Frequency(dataframe, column_name,bins, path, file_name):
    
    # Create a new column with the numerical variable cut in bins
    dataframe['Band_{col}'.format(col = column_name)] = pd.cut(dataframe[column_name],
                                                               bins,
                                                               right = False)
    
    # Calculate Frequency
    frequency = pd.DataFrame(dataframe['Band_{col}'.format(col = column_name)].value_counts())
    
    # Export Frequency
    frequency.to_csv(os.path.join(path, file_name))




# Define a function to create regular crosstab
def Crosstab(dataframe, column_1, column_2, path, file_name):
    
    # Create a crosstab
    crosstab = pd.crosstab(dataframe[column_1],
                           dataframe[column_2],
                           margins = 'Total',
                           dropna = False)
    
    # Export Crosstab
    crosstab.to_csv(os.path.join(path, file_name))




# Banded Crosstab
def Banded_Crosstab(dataframe, column_1, column_2, bins, path, file_name):
    
    # Create Banded columns for each numerical variable
    dataframe['Band_{col_1}'.format(col_1 = column_1)] = pd.cut(dataframe[column_1], bins, right = False)
    
    dataframe['Band_{col_2}'.format(col_2 = column_2)] = pd.cut(dataframe[column_2], bins, right = False)
    
    
    # Create Crosstab
    crosstab = pd.crosstab(dataframe['Band_{col_1}'.format(col_1 = column_1)],
                           dataframe['Band_{col_2}'.format(col_2 = column_2)],
                           margins = 'Total',
                           dropna = False)
    
    # Export Crosstab
    crosstab.to_csv(os.path.join(path, file_name))



# Define a function for exporting Pivot tables
def Pivot_Table(dataframe, values_col, index_col, columns_col, function, path, file_name):
    
    # Create a pivot table
    pivot_table = pd.DataFrame(pd.pivot_table(dataframe,
                                 values = values_col,
                                 index = index_col,
                                 columns = columns_col,
                                 aggfunc = function,
                                 fill_value=(0)))
    
    # Export Pivot table
    pivot_table.to_csv(os.path.join(path, file_name))



# Define a function to create Pie Chart
def Pie_Chart(dataframe, variable, path, file_name):
    
    # Create Plot
    plot = dataframe[variable].value_counts().plot(kind = 'pie')
    
    chart = plot.figure
    
    # Export Plot
    chart.savefig(os.path.join(path, file_name))



# Define a function to create Bar Chart
def Bar_Chart(dataframe, variable, title, path, file_name):
    
    # Create Plot
    plot = dataframe[variable].value_counts().plot(kind = 'bar')
    
    # Add title
    plot.set_title(title)
    
    chart = plot.figure
    
    # Export Plot
    chart.savefig(os.path.join(path, file_name))



# Define a function to create Scatter Plot
def Scatter_Plot(dataframe, variable_1, variable_2, title, path, file_name):
    
    # Create Plot
    plot = dataframe.plot(kind = 'scatter', x = variable_1, y = variable_2)
    
    # Add title
    plot.set_title(title)
    
    chart = plot.figure
    
    # Export Plot
    chart.savefig(os.path.join(path, file_name))


# Density Plot
def Density_Plot(dataframe, variable,title, path, file_name):
    
    # Create Plot
    plot = dataframe[variable].plot(kind = 'density')
    
    # Calculate Mean
    plot.axvline(dataframe[variable].mean(), color='red')
    
    # Calculate Median
    plot.axvline(dataframe[variable].median(), color='green')
    
    # Add title
    plot.set_title(title)
    
    chart = plot.figure
    
    # Export Plot
    chart.savefig(os.path.join(path, file_name))


# Good Luck!