#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 18:53:52 2023

@author: juliana
"""


'''
Machine Learning Model Interpretation with SHAP

on BankCaseStudy
'''



# Preliminaries
data_path = r"/home/juliana/Документи/Experian - ML Study/Data"

reports_path = r"/home/juliana/Документи/Experian - ML Study/Reports/SHAP Exploration"

data_file = r"BankCaseStudyData.csv"




# Import Python APIs
import numpy as np
import os
import pandas as pd
import shap
import lightgbm
import graphviz
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt



# Read Data
data_all = pd.read_csv(os.path.join(data_path, data_file),
                       keep_default_na = False,
                       low_memory = False,
                       encoding = 'utf8')



# Derive Goods and Bads
data_gb = data_all[data_all['GB_Flag'].isin(['Good', 'Bad'])].copy()



# Create Train Test split
data_dev = data_gb[data_gb['split'].isin(['Development'])].copy()

data_val = data_gb[data_gb['split'].isin(['Validation'])].copy()




# Data Processing
columns = data_dev.columns


# Insurance required
# =============================================================================
# crosstab_1 = pd.crosstab(data_dev['Insurance_Required'],
#                          data_dev['GB_Flag'])
# =============================================================================

data_dev['Insurance_Required'] = data_dev['Insurance_Required'].map({'Y' : 1,
                                                                     'N' : 2,
                                                                     '' : 0})



data_val['Insurance_Required'] = data_val['Insurance_Required'].map({'Y' : 1,
                                                                     'N' : 2,
                                                                     '' : 0})



# Loan Payment Frequency
# =============================================================================
# crosstab_2 = pd.crosstab(data_dev['Loan_Payment_Frequency'],
#                          data_dev['GB_Flag'])
# =============================================================================


data_dev['Loan_Payment_Frequency'] = data_dev['Loan_Payment_Frequency'].map({'' : 0,
                                                                             'X' : 0,
                                                                             'F' : 0,
                                                                             'W' : 1,
                                                                             'M' : 2})


data_val['Loan_Payment_Frequency'] = data_val['Loan_Payment_Frequency'].map({'' : 0,
                                                                             'X' : 0,
                                                                             'F' : 0,
                                                                             'W' : 1,
                                                                             'M' : 2})



# Marital Status
# =============================================================================
# crosstab_3 = pd.crosstab(data_dev['Marital_Status'],
#                          data_dev['GB_Flag'])
# =============================================================================


data_dev['Marital_Status'] = data_dev['Marital_Status'].map({'D' : 0,
                                                             'W' : 0,
                                                             'Z' : 0,
                                                             'S' : 1,
                                                             'M' : 2})


data_val['Marital_Status'] = data_val['Marital_Status'].map({'D' : 0,
                                                             'W' : 0,
                                                             'Z' : 0,
                                                             'S' : 1,
                                                             'M' : 2})


# Residential Status
# =============================================================================
# crosstab_4 = pd.crosstab(data_dev['Residential_Status'],
#                          data_dev['GB_Flag'])
# =============================================================================


data_dev['Residential_Status'] = data_dev['Residential_Status'].map({'L' : 0,
                                                                     'T' : 1,
                                                                     'O' : 2,
                                                                     'H' : 3})


data_val['Residential_Status'] = data_val['Residential_Status'].map({'L' : 0,
                                                                     'T' : 1,
                                                                     'O' : 2,
                                                                     'H' : 3})




# Occupation_Code
# =============================================================================
# crosstab_5 = pd.crosstab(data_dev['Occupation_Code'],
#                          data_dev['GB_Flag'])
# =============================================================================


data_dev['Occupation_Code'] = data_dev['Occupation_Code'].map({'O' : 0,
                                                               'B' : 1,
                                                               'P' : 2,
                                                               'M' : 3})


data_val['Occupation_Code'] = data_val['Occupation_Code'].map({'O' : 0,
                                                               'B' : 1,
                                                               'P' : 2,
                                                               'M' : 3})



# Model features
features_list = ['Gross_Annual_Income',
                     'Loan_Amount', 
                     'Number_of_Dependants',
                     'Number_of_Payments',
                     'Time_at_Address',
                     'Time_in_Employment',
                     'Time_with_Bank',
                     'SP_ER_Reference',
                     'SP_Number_of_CCJs',
                     'SP_Number_Of_Searches_L6M',
                     'Insurance_Required',
                     'Loan_Payment_Frequency',
                     'Marital_Status',
                     'Residential_Status',
                     'Occupation_Code'
                     ]


features = data_dev[features_list].copy()

features = features.fillna(-1)


# Model Target
target = data_dev[['GB_Flag']].copy()





# Model development
model = LGBMClassifier(num_leaves = 10,
                       max_depth = 5,
                       n_estimators = 10,
                       min_child_samples = 800)



# Fit model to the data
model.fit(features,
          target)



# List model features
model_features = model.feature_name_


# Calculate score
val_features = data_val[model_features].copy()

data_val['ML Score'] = (model.predict(val_features, raw_score = True) * 100).round(decimals = 0)

data_dev['ML Score'] = (model.predict(features, raw_score = True))



# ML Outcome
data_val['ML Pred'] = model.predict(val_features)

data_dev['ML Pred'] = model.predict(features)



# Accuracy
print(accuracy_score(data_dev['GB_Flag'], data_dev['ML Pred']))

print(accuracy_score(data_val['GB_Flag'], data_val['ML Pred']))





''' SHAP '''


# Explainer
explainer = shap.TreeExplainer(model, model_output = 'probability',
                               feature_perturbation = 'interventional',
                               data = features)


shap_values = explainer.shap_values(features)

print(explainer.expected_value)



# Calculate ML Score from Probability
ml_probability = pd.DataFrame(model.predict_proba(features))

ml_probability = ml_probability.rename(columns = {0 : 'Bad', 1 : 'Good'})



# Scale SHAP
shap_results = pd.DataFrame(shap_values)

shap_results = shap_results.rename(columns = {0 : 'SHAP_Gross_Annual_Income',
                     1 : 'SHAP_Loan_Amount', 
                     2 : 'SHAP_Number_of_Dependants',
                     3 : 'SHAP_Number_of_Payments',
                     4 : 'SHAP_Time_at_Address',
                     5 : 'SHAP_Time_in_Employment',
                     6 : 'SHAP_Time_with_Bank',
                     7 : 'SHAP_SP_ER_Reference',
                     8 : 'SHAP_SP_Number_of_CCJs',
                     9 : 'SHAP_SP_Number_Of_Searches_L6M',
                     10 : 'SHAP_Insurance_Required',
                     11 : 'SHAP_Loan_Payment_Frequency',
                     12 : 'SHAP_Marital_Status',
                     13 : 'SHAP_Residential_Status',
                     14 : 'SHAP_Occupation_Code'
                     })


shap_results['Model Expected Value'] = explainer.expected_value


shap_results['Sum of SHAP and Expected Value'] = shap_results.sum(axis = 1)


shap_results['ML Score'] = ml_probability['Good']

shap_results = shap_results * 1000


# =============================================================================
# example_unscaled = shap_results[: 1000]
# 
# example_scaled = shap_results[: 1000]
# 
# =============================================================================



# Summary Plots

shap_results_all = pd.concat([shap_results, features, data_dev['GB_Flag']], axis = 1, join = 'outer')

shap_results_all = shap_results_all.dropna()



# Average feature contribution
shap_features = shap_results.drop(['Model Expected Value', 'Sum of SHAP and Expected Value', 'ML Score'], axis = 1)

shap_features = pd.DataFrame({'SHAP' : shap_features.abs().mean()})

shap_features = shap_features.reset_index()

shap_bar_plot = sns.barplot(shap_features, x = 'index', y = 'SHAP')

shap_bar_plot.set_xticklabels(model_features,rotation=90)



# Histogram of SHAP
shap_hist = sns.histplot(shap_results,
                         x = 'SHAP_Gross_Annual_Income')


# Partial Dependence Plot
shap_partial_dependence = sns.scatterplot(shap_results_all,
                                          x = 'Time_with_Bank',
                                          y = 'SHAP_Time_with_Bank',
                                          hue = 'GB_Flag')




# =============================================================================
# # Good Class
# shap_good = shap_values[1]
# 
# =============================================================================


# For a Single record
first_record = pd.DataFrame(features.take([3]))

shap_values_first_record = explainer.shap_values(first_record)

shap_good_first_record = pd.DataFrame(shap_values_first_record)

shap_good_first_record = shap_good_first_record.rename(columns = {0 : 'Gross_Annual_Income',
                     1 : 'Loan_Amount', 
                     2 : 'Number_of_Dependants',
                     3 : 'Number_of_Payments',
                     4 : 'Time_at_Address',
                     5 : 'Time_in_Employment',
                     6 : 'Time_with_Bank',
                     7 : 'SP_ER_Reference',
                     8 : 'SP_Number_of_CCJs',
                     9 : 'SP_Number_Of_Searches_L6M',
                     10 : 'Insurance_Required',
                     11 : 'Loan_Payment_Frequency',
                     12 : 'Marital_Status',
                     13 : 'Residential_Status',
                     14 : 'Occupation_Code'
                     })


shap_good_first_record['Model Expected Value'] = explainer.expected_value

shap_good_first_record['ML Score'] = shap_good_first_record.sum(axis = 1)


shap_good_first_record = shap_good_first_record * 1000

shap_plot_first_record = sns.barplot(shap_good_first_record)

shap_plot_first_record.set_xticklabels(model_features,rotation=90)





# Summary Stats
gb_flag = data_all['GB_Flag'].value_counts(dropna = False).sort_index()










# =============================================================================
# # Summary Plot
# shap.summary_plot(shap_good, 
#                   features,
#                   model_features, link = 'logit')
# 
# =============================================================================

# =============================================================================
# # Force Plot
# shap.force_plot(explainer.expected_value, shap_values[0], features.iloc[[0]], feature_names=model_features,
#                 matplotlib=True,show=False)
# 
# plt.savefig(os.path.join(reports_path, r"SHAP Force Plot.png"), dpi=150, bbox_inches='tight')
# 
# =============================================================================


# Visualize Tree
graph = lightgbm.create_tree_digraph(model, tree_index = 0)

graph.render(os.path.join(reports_path, r"LightGBM_Model"), format = 'png')



# Good Luck!