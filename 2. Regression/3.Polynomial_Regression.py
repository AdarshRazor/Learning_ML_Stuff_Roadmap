# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 00:52:04 2021

@author: AdarshRazor
"""
# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('3.PR_Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # taking independent variable and we want feature in form of matrics. That's why putting it as 1:2
y = dataset.iloc[:, 2].values   # taking dependent variable
#Remove the comment to see the graph is plotting in polunomial way hence we will use ploynomial regression to understand the realtion
#plt.scatter(X, y, color = 'red')

# Splitting the dataset into the Training set and Test set
# Its better not to split the data as we have only 10 test case for the training.
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

'''
We make both Linear Regression as well as Polynomial Regression to comapre between the 2 graph
'''
#  Linear Regression 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Polynomial Regression 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 9) # degreee of the polynomial features
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
pred_test1=lin_reg.predict([[6.5]]) # check in the vairable explorer

# Predicting a new result with Polynomial Regression
pred_test2=lin_reg_2.predict(poly_reg.fit_transform([[6.5]])) # check in the vairable explorer
