'''
Lab by -> Thomas Haskell

Topic -> Classification with Decision Trees
Source -> IBM Machine Learning with Python Certification Course

Dataset -> An example of a multiclass classification, patients with the same
illness all of which responded to one of 5 drugs most effectively. Features of 
the dataset are Age, Sex, Blood Pressure, and Cholesterol. The target is one of
the 5 testing drugs Drug A, Drug B, Drug c, Drug x and y.

Objectives:
1. Develop a classification model using Decision Tree Algorithms
2. Visualize data and gain familiarity with pandas and numpy libraries
'''
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree


## Downloading the dataset ##
import urllib.request
import ssl
ssl._create_default_https_context = ssl._create_unverified_context  #workaround for SSL certificate verification
def download(url, filename):
    urllib.request.urlretrieve(url, filename)
    print("Download Complete")
path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
download(path, "drug200.csv")
my_data = pd.read_csv("drug200.csv")  #reading dataset into dataframe
print(my_data.shape)
print(my_data[0:5])

## Pre-Processing ##
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values  #target column removed, not numerical
print(X[0:5])




