# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 19:27:59 2021

@author: AdarshRazor
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('2.MLR_50_Startups.csv')
X = dataset.iloc[:, :-1].values # selecting the independent variable
y = dataset.iloc[:, 4].values # selecting the dependent variable

# Encoding categorical data
'''
LableEncoder is used to label the variable 
OneHotEncoder is used to create dummy variable
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3]) # We are taking the third column and labelling it into numbers
onehotencoder = OneHotEncoder(categorical_features = [3]) # Here we are creating dummy variable from the third column
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:] #Always take 1 less dummmy variable

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling will be applied by our multiple linear regression model

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

