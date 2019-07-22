# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

titanic_df = pd.read_csv("D:\\Git\\MachineLearning\\ScikitLearn\\Titanic\\dataset\\new_train.csv")

features = list(titanic_df.columns[1:])

result_dict={}

#Summarizing output
def summarize_classification(y_test, y_pred):
    
    acc = accuracy_score(y_test, y_pred, normalize=True)
    num_acc =accuracy_score(y_test, y_pred, normalize=False)
    
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test,y_pred)
    
    return ("accuracy" : acc,
            "precission" : prec,
            "recall" : recall,
            "accuracy_count" : numm_acc)

#Helper function for building models
def build_model(classifier_fn,
                name_of_x_col,
                name_of_y_col,
                dataset,
                test_frac=0.2):
    
    X = dataset[name_of_x_col]
    Y = dataset[name_of_y_col]
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_frac)
    
    model = classifier_fn(x_train, y_train)
    
    y_pred = model.predict(y_test)
    
    y_pred_train = model.predict(y_train)
    
    train_summary = summarize_classification(y_train, y_pred_train)
    test_summary = summarize_classification(y_test, y_pred)
    
    pred_result = pd.DataFrame({"y_test":y_test,
                                "y_pred":y_pred})
    
    model.crosstab = pd.crosstab(pred_result.y_pred, pred_result.y_test)
    
    return {"training": train_summary,
            "test": test_summary,
            "confusion_matrix": model.crosstab}