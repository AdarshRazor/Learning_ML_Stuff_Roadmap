# -*- coding: utf-8 -*-
"""
Adarsh Razor

06-06-2021 01:22AM
"""

#importing the libraries
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
'''
Here we will be predicting the salary of the person based on the year of experience
'''
dataset = pd.read_csv('1.SLR_Salary_Data.csv')
X = dataset.iloc[:, :-1].values #independent variable (Years of experience)
Y = dataset.iloc[:, 1].values  #dependent variable (Salary)

#splitting data into testing and training set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0 )
'''
We will train our model with X_train and Y_train 
Then will test it out with X_test and Y_test and check if the results are coming as per our expectations or not.
'''

from sklearn.linear_model import LinearRegression
regressor = LinearRegression() # Our linear regression model
regressor.fit(X_train, Y_train) # training data is ready 

y_pred = regressor.predict(X_test)
'''
Here we putting X_test because X is independent variable and on basis of X, Y will be getting predicted
'''
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.show()

plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue') # comparing here with the test set
plt.show()