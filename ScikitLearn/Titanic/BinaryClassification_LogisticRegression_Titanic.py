# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 11:45:05 2019

@author: sarthak
"""

import pandas as pd 

titanic_df = pd.read_csv("C:\\Users\\sarth\\Documents\\ScikitLearn\\Datasets\\titanic\\new_train.csv")

from sklearn.model_selection import train_test_split

X = titanic_df.drop("Survived", axis=1)
Y = titanic_df["Survived"]

x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size=0.2)

x_train.shape
x_test.shape

from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(penalty="l2", C=1, solver="liblinear").fit(x_train,y_train)

y_pred = logistic_model.predict(x_test)

#Setup a test vs predicted dataset
pred_result = pd.DataFrame ({"y_test":y_test,"y_pred":y_pred})

#Confusion matrix
titanc_crosstab = pd.crosstab(pred_result.y_test, pred_result.y_pred)

#Passing the predicted result through metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)