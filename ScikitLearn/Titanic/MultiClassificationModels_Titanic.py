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
    
    return {"accuracy" : acc,
            "precission" : prec,
            "recall" : recall,
            "accuracy_count" : num_acc}

#Helper function for building models
def build_model(classifier_fn,
                name_of_y_col,
                name_of_x_col,
                dataset,
                test_frac=0.2):
    
    X = dataset[name_of_x_col]
    Y = dataset[name_of_y_col]
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_frac)
    
    model = classifier_fn(x_train, y_train)
    
    y_pred = model.predict(x_test)
    
    y_pred_train = model.predict(x_train)
    
    train_summary = summarize_classification(y_train, y_pred_train)
    test_summary = summarize_classification(y_test, y_pred)
    
    pred_result = pd.DataFrame({"y_test":y_test,
                                "y_pred":y_pred})
    
    model.crosstab = pd.crosstab(pred_result.y_pred, pred_result.y_test)
    
    return {"training": train_summary,
            "test": test_summary,
            "confusion_matrix": model.crosstab}

#Helper function to compare results
def compare_results():
    for key in result_dict:
        print("classification" + key)
        
        print()
        print("Training Data")
        for score in result_dict[key]["training"]:
            print(score, result_dict[key]["training"][score])
           
        print()
        print("Test Data")
        for score in result_dict[key]["test"]:
            print(score, result_dict[key]["test"][score])
            
        print()

#Creating regression models
#Logistic Regression
def logistic_fn(x_train,y_train):
    model = LogisticRegression(solver="liblinear")
    model.fit(x_train, y_train)
    
    return model

result_dict["survived ~ logisitic"] = build_model(logistic_fn,
                                                  'Survived',
                                                  features,
                                                  titanic_df)

#Linear Discriminant Analysis (underlying PCA)
def linear_discriminant_analysis(x_train,y_train,solver="svd"):
    model = LinearDiscriminantAnalysis(solver=solver)
    model.fit(x_train, y_train)
    
    return model

result_dict["survived ~ linear_discriminant_analysis"] = build_model(linear_discriminant_analysis,
                                                                     'Survived',
                                                                     features[0:-1], #use dummy encoding by dropping one of the hot encoding columns to avoid variable collinearity  
                                                                     titanic_df)

#Quadratic Discriminant Analysis
def quadratic_discriminant_analysis(x_train, y_train):
    model = QuadraticDiscriminantAnalysis()
    model.fit(x_train,y_train)
    
    return model

result_dict["survived ~ quadratic_disriminant_analysis"] = build_model(quadratic_discriminant_analysis,
                                                                       "Survived",
                                                                       features[0:-1],
                                                                       titanic_df)

def sgd_fn(x_train, y_train, max_iter=100, tol=1e-3):
    model=SGDClassifier(max_iter=max_iter, tol=tol)
    model.fit(x_train, y_train)
    
    return model 

result_dict["survived ~ Stochastic Gradient Classifier"] = build_model(sgd_fn,
                                                                       "Survived",
                                                                       features,
                                                                       titanic_df)

compare_results()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        