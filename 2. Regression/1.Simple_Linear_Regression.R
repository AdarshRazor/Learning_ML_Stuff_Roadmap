"
Adarsh Razor

06-06-2021 02:10AM
"

#importing the dataset
dataset = read.csv('1.SLR_Salary_Data.csv') 

#importing libraries to split the data
#install.packages('caTools')
library(caTools) #this will activate the library to be used in the script

set.seed(123) #shuffle applied to data just like random_State in python

"
Herer we are splitting the data and the split ratio is for training data instead of testing data
The training set are marked to be true as we denoted it as TRUE
The training set are marked to be true as we denoted it as FALSE
"
split = sample.split(dataset$Salary, SplitRatio = 2/3) 
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

#Now we fitting our linear model and the formula
regressor = lm(formula = Salary ~ YearsExperience, data = training_set)

#Predicting the test set results
Y_pred = predict(regressor, newdata = test_set)

#Visualizing the dataset
#install.packages('ggplot2')
library(ggplot2)

'
Incomplete code 
connect at adarashanshu7@gmail.com for the codes and explanation
'