#Adarsh Razor

#06-06-2021 02:10AM

#importing the dataset
dataset = read.csv('1.SLR_Salary_Data.csv') 

#importing libraries to split the data
#install.packages('caTools')
library(caTools) #this will activate the library to be used in the script

set.seed(123)

split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)