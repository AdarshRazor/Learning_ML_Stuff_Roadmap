# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 15:25:38 2021

@author: AdarshRazor
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('3.PR_Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # puuting features as matrics 
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
# Again we are not splitting the dataset because of low training set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1)) # reshape if your data has a single feature

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') # we are taking gaussian kernel which is 'rbf' one
regressor.fit(X, y)

# Predicting a new result
#In the below code we are transforming 6.5 as we applied the same to X and Y so it fit the data.
''' Press ctrl + I to know more about any funtion'''
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]]))) # tranform method expect array and that's why we changing it by np.array
y_pred = sc_y.inverse_transform(y_pred)

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()