# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load Data: Read the dataset using pd.read_csv() and display first and last few rows.

2.Prepare Data: Separate features (hours) and target variable (scores) for training and testing.

3.Split Data: Use train_test_split() to divide the dataset into training and testing sets.

4.Train Model: Fit a linear regression model using the training data.

5.Evaluate and Plot: Predict scores on the test set, and visualize results with scatter and line plots.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SURYAMALAR V
RegisterNumber: 21223230224
*/
```
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
```
```
dataset = pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
```
![Screenshot 2024-08-29 110503](https://github.com/user-attachments/assets/2317044c-b221-42bf-9ba9-a7274f3242c5)
```
dataset.info()
```
![image](https://github.com/user-attachments/assets/aac188f0-ff67-4096-af4e-b3dd766d5c14)
```
#assigning hours to X & scores to Y
X = dataset.iloc[:,:-1].values
print(X)
Y = dataset.iloc[:,-1].values
print(Y)
```
![image](https://github.com/user-attachments/assets/d5f60098-ffcc-46b0-a34e-0b098c6c2c69)
```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
```
```
X_train.shape,X_test.shape
```
![image](https://github.com/user-attachments/assets/becfaffe-b1f9-4a1e-8166-0d55d1a4732d)
```
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,Y_train)
```
![image](https://github.com/user-attachments/assets/b390a07a-ab77-4a19-bdb8-7c562f82fdb9)
```
Y_pred = reg.predict(X_test)
print(Y_pred)
print(Y_test)
```
![image](https://github.com/user-attachments/assets/90757fd0-171a-40ab-b5e5-2396850e196e)
```
plt.scatter(X_train,Y_train,color="green")
plt.plot(X_train,reg.predict(X_train),color="red")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show
```
![image](https://github.com/user-attachments/assets/5a468b76-cb32-42f3-a07d-790c0eea3932)
```
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,reg.predict(X_test),color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show
```
![image](https://github.com/user-attachments/assets/62b3545e-c6a4-414b-a1ef-d7fc529b647c)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
