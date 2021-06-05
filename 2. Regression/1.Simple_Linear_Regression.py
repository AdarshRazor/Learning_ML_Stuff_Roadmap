# -*- coding: utf-8 -*-
"""
Adarsh Razor

06-06-2021 01:22AM
"""

#importing the libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('1.SLR_Salary_Data.csv')
X = dataset.iloc[:, :-1].values #independent variable
Y = dataset.iloc[:, 1].values  #dependent variable

#splitting data into testing and training set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0 )
































