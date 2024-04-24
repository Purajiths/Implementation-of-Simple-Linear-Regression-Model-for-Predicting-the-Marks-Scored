# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1 Import necessary libraries (e.g., pandas, numpy,matplotlib)

2 Load the dataset and then split the dataset into training and testing sets using sklearn library.

3 Create a Linear Regression model and train the model using the training data (study hours as input, marks scored as output).

4 Use the trained model to predict marks based on study hours in the test dataset.

5 Plot the regression line on a scatter plot to visualize the relationship between study hours and marks scored.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 
RegisterNumber: 
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/MLSET.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='orange')
lr.coef_
lr.intercept_
```
## Output:
## 1)HEAD:
![image](https://github.com/Purajiths/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145548193/18f60b8b-87b0-4ec7-807d-d7137f40d42b)


## 2)GRAPH OF PLOTTED DATA:
![image](https://github.com/Purajiths/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145548193/d9dbd099-0f86-4b9c-ae6a-9d6f1911798d)

## 3)TRAINED DATA:
![image](https://github.com/Purajiths/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145548193/768a2fca-818d-4087-9646-444141bdf820)


## 4)LINE OF REGRESSION:
![image](https://github.com/Purajiths/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145548193/e7ae99a4-c5cd-4eb1-a946-5677745b5c78)


## 5)COEFFICIENT AND INTERCEPT VALUES:
![image](https://github.com/Purajiths/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145548193/87fae004-c5ca-4fa6-960a-a1b672e5ef2d)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
