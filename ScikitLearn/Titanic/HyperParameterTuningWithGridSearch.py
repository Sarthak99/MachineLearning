# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 12:04:02 2019

@author: sarthak
"""

import pandas as pd
#import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

titanic_df = pd.read_csv("D:\\Git\\MachineLearning\\ScikitLearn\\Titanic\\dataset\\new_train.csv")

X = titanic_df.drop("Survived", axis=1)
Y = titanic_df["Survived"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

#Summarizing output
def summarize_classification(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred, normalize=True)
    num_acc =accuracy_score(y_test, y_pred, normalize=False)
    
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test,y_pred)
    
    print()
    print("Test Data Count:", len(y_test))
    print("Accuracy_count:", num_acc)
    print("accuracy_score:", acc)
    print("precison:", prec)
    print("Recall:", recall)
    print()

from sklearn.model_selection import GridSearchCV
# =============================================================================
# We are going to tune the DecisionTree classifier with the hyperparameter = "max_depth".
# The goal of this code is to find the best value of "max_depth" for giving the most accurate result 
# =============================================================================

parameter = {"max_depth":[2,4,6,8,10]}

grid_search = GridSearchCV(DecisionTreeClassifier(), parameter, cv=3, return_train_score=True)

# =============================================================================
# GridSearch will create a grid of all the parameters being considered and create that many models.
# Each model will compared and the best value result can be achieved with a metrics score.
# For eg: If there is 1 hyperparameter being tuned with 6 different values, then the no.of models generated will be 6 models.
# If there are 2 hyperparameter being tuned with 6 different values each, then the no.of models generated will be
# 6*6 = 36 models.
# GridSearchCV
#    1st: classifier
#    2nd: hyperParameter
#    3rd: "cv=3" --> This means the dataset will be divided into 3 sets for cross validation.
#                    It uses 1/3 of the set as a test data in training phase and rather learns on the 2/3 of the train set
#    4th: This is the default scoring mechanism i.e accuracy that is used to compare the different models.
# =============================================================================

grid_search.fit(x_train,y_train)

#Belowcommand returns the parameter that performs the best
grid_search.best_params_


# =============================================================================
# Below loop can be used to access the content and scores of all the models and find out 
# the score assigned for each model
# =============================================================================

for i in range(5):
    print("Parameter:", grid_search.cv_results_["params"][i])
    print("Mean Test score:", grid_search.cv_results_["mean_test_score"][i])
    print("Rank:", grid_search.cv_results_["rank_test_score"][i])
    

decision_tree = DecisionTreeClassifier(max_depth=grid_search.best_params_["max_depth"]).fit(x_train, y_train)
y_pred = decision_tree.predict(x_test)

summarize_classification(y_test, y_pred)

# =============================================================================
# Now to apply tuning into multiple hyperparameters. We will LogisticRegression classifier and hyperparameters would be the penalty function
# and the regularization strengths
# =============================================================================

parameters = {"penalty":["l1","l2"],"C":[0.1,0.4,0.8,1,2,5]}

# =============================================================================
# Number of models that will trained here will be 2*6=12 models
# =============================================================================

grid_search=GridSearchCV(LogisticRegression(solver="liblinear"), parameters, cv=3,return_train_score=True)

grid_search.fit(x_train, y_train)

for i in range(12):
    print("Parameter:", grid_search.cv_results_["params"][i])
    print("Mean Test score:", grid_search.cv_results_["mean_test_score"][i])
    print("Rank:", grid_search.cv_results_["rank_test_score"][i])


log_reg = LogisticRegression(solver = "liblinear", penalty = grid_search.best_params_["penalty"],C = grid_search.best_params_["C"]).fit(x_train, y_train)
y_pred = log_reg.predict(x_test)

summarize_classification(y_test, y_pred)
