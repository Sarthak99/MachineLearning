# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 12:43:24 2019

@author: sarth
"""

import sklearn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

titanic_df = pd.read_csv("C:\\Users\\sarth\\Documents\\ScikitLearn\\Datasets\\titanic\\train.csv")

titanic_df.shape

new_titanic_df = titanic_df.drop(["PassengerId","Name","Ticket","Cabin"], "columns")

new_titanic_df.shape

new_titanic_df[new_titanic_df.isnull().any(axis=1)].count()
new_titanic_df = new_titanic_df.dropna()

new_titanic_df.describe()

#plots of labels against survival rate

fig, ax = plt.subplots(figsize=(12,8))
plt.scatter(new_titanic_df["Age"],new_titanic_df["Survived"])
plt.xlabel("Age")
plt.ylabel("Survived")
plt.title("Age v/s Survival relationship")

fig, ax = plt.subplots(figsize=(12,8))
plt.scatter(new_titanic_df["Sex"],new_titanic_df["Survived"])
plt.xlabel("Sex")
plt.ylabel("Survived")
plt.title("Sex v/s Survival relationship")

fig, ax = plt.subplots(figsize=(12,8))
plt.scatter(new_titanic_df["Fare"],new_titanic_df["Survived"])
plt.xlabel("Fare")
plt.ylabel("Survived")
plt.title("Fare v/s Survival relationship") 
#This plot shows a fair relationship that passengers that had paid higher fair had a better survival rate

#Now to show the info in a matrix format,pd.crosstab can be used.

pd.crosstab(new_titanic_df["Sex"],new_titanic_df["Survived"])

pd.crosstab(new_titanic_df["Pclass"],new_titanic_df["Survived"])

# =============================================================================
# Survived    0    1
# Pclass            
# 1          64  120
# 2          90   83
# 3         270   85
#This shows a clear relation that passengers in a higher passenger class were given a more preferable 
#survival chance than lower class passengers
# =============================================================================

#To find the correlation between different labels, we can use .corr() method on the DF

new_titanic_df_corr = new_titanic_df.corr()

#use a heatmap to plot the correlation obtained
fig, ax = plt.subplot(figsize=(12,8))
sns.heatmap(new_titanic_df_corr, annot=True)

#Preprocessing and shuffling of data to new csv
from sklearn import preprocessing
label_encoding = preprocessing.LabelEncoder()
#Convert the binary values in Sex column to numeric codes
new_titanic_df["Sex"] = label_encoding.fit_transform(new_titanic_df["Sex"].astype(str))

label_encoding.classes_
#array(['female', 'male'], dtype=object)

#one-hot encoding of column values
new_titanic_df = pd.get_dummies(new_titanic_df, columns=["Embarked"])
new_titanic_df.head()
# =============================================================================
#    Survived  Pclass  Sex   Age  ...     Fare  Embarked_C  Embarked_Q  Embarked_S
# 0         0       3    1  22.0  ...   7.2500           0           0           1
# 1         1       1    0  38.0  ...  71.2833           1           0           0
# =============================================================================

#Shuffling the data and dropping old indices
new_titanic_df = new_titanic_df.sample(frac=1).reset_index(drop=True)

new_titanic_df.to_csv("C:\\Users\\sarth\\Documents\\ScikitLearn\\Datasets\\titanic\\new_train.csv",index=False)
