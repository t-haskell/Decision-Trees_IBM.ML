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
from sklearn import preprocessing
# converts sex to numerical values
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:,1] = le_sex.transform(X[:,1])
# convert blood pressure to numerical values
le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])
# convert cholesterol to numerical values
le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3])
print(X[0:5])
# now filling target variable
y = my_data["Drug"]
y[0:5]

## Setting up Decision Tree ##
from sklearn.model_selection import train_test_split
# Test/Train Split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
print('Shape of X training set{}'.format(X_trainset.shape),'&', ' Size of Y training set {}'.format(y_trainset.shape))
print('Shape of X testing set{}'.format(X_testset.shape),'&', ' Size of Y testing set {}'.format(y_testset.shape))
# Modeling
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
print(drugTree) #default parameters
drugTree.fit(X_trainset, y_trainset)

## Predicting ##
predTree = drugTree.predict(X_testset)
print(predTree[0:5])
print(y_testset[0:5])

## Evaluation ##
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTree's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

## Visualization ##
tree.plot_tree(drugTree)
plt.show()