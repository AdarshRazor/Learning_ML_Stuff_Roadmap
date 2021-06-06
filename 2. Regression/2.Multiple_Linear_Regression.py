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
from sklearn.compose import ColumnTransformer
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3]) # We are taking the third column and labelling it into numbers
onehotencoder = ColumnTransformer([("State",    #name of the column
                                    OneHotEncoder(),    # the transformer class
                                    [3])],      # column to be applied on
                                      remainder='passthrough')      # donot apply anything to the remaining columns [ Kinda very important parameter ]
X = np.array(onehotencoder.fit_transform(X))

# Avoiding the Dummy Variable Trap
X = X[:, 1:] #Always take 1 less dummmy variable

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling will be applied by our multiple linear regression model

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Multiple linear regression cannot be plot here since it contain too many vaiables, here is an example why (Just remove the quotes and run the code)
'''
plt.scatter(X_train[:, 2], Y_train, color='red')
plt.plot(X_train[:, 2], regressor.predict(X_train), color='blue')
plt.show()
'''
