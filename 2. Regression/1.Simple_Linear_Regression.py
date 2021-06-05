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
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values