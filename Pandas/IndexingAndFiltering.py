import pandas as pd
import numpy as np
df = pd.read_pickle("C:\\Users\\sarth\\Documents\\Pandas\\Dataset\\collection-master\\data_frame1.pickle")
artists = df['artist']
pd.unique(artists)
print(len(pd.unique(artists)))

#Find the no.of times the artist exists in the DataFrame
s = df['artist'] == "Bacon, Francis"
print(s.value_counts())

#Using value_counts() directly on the DataFrame
artist_count = df['artist'].value_counts()
print(artist_count['Bacon, Francis'])

artist_count

#usage of loc and iloc
print(df.loc[1035,"artist"])
print(df.iloc[0,0])

print(df.iloc[0,:])

print(df.iloc[0:2,0:2])

#Find the area of each artwork
df["height"]*df["width"]

#above error is due to the columns width or height having non-numeric values; values in a series are always treated as objects
#convert object to numeric by below
pd.to_numeric(df["width"])  #this also fails as there are some values that cannot be converted to numeric like Strings

print(df.iloc[1839,:])

#We need force such values to NaN (Not a number)
pd.to_numeric(df["width"], errors="coerce")

print(df.iloc[0,:])

#Feed the converted result back into the DataFrame
df.loc[:,"width"] = pd.to_numeric(df["width"],errors="coerce")

print(df.iloc[0,:])

df["width"].sort_values().head()
df["width"].sort_values().tail()

#similarly convert the height column to all float
df.loc[:,"height"] = pd.to_numeric(df["height"], errors = "coerce")

df["height"].sort_values().head()

print(df["height"].dtype)

#Create a new column "area" in the DF
area = df["width"]*df["height"]
df = df.assign(area=area)

df["area"].max()

#find the index that has maxvalue
df["area"].idxmax()

df.loc[98367,:]

#Groups and Iteration 
small_df = df.loc[6745:6755,:]

grouped  = small_df.groupby("artist")

type(grouped)

#Iteration over the group
for name, group_df in grouped:
    print(name)
    print(group_df)
    
#Aggregate
#Min function
for name,group_df in small_df.groupby("artist"):
    min_year = group_df["acquisitionYear"].min()
    print("{}: {}".format(name, min_year))
    break

#Transformation
#lets change few values in medium column to Nan and fill them with the most frequent value
def fill_values(series):
    print(series)
    values_counted = series.value_counts()
    if values_counted.empty:
        return series
    most_frequent = values_counted.index[0]
    new_medium = series.fillna(most_frequent)
    return new_medium

def transform_df(source_df):
    group_dfs = []
    for name,group_df in source_df.groupby("artist"):
        filled_df = group_df.copy()
        filled_df.loc[:,"medium"] = fill_values(group_df["medium"])
        group_dfs.append(filled_df)   
    new_df = pd.concat(group_dfs)
    return new_df 

#Now check the result
filled_df = transform_df(small_df)
filled_df.loc[:,"medium"]

#BUILT_IN METHODS

#Transform
grouped_medium = small_df.groupby("artist")["medium"]
small_df.loc[:,"medium"]=grouped_medium.transform(fill_values)

#Aggregate -- min
df.groupby("artist").agg(np.min)
df.groupby("artist").min()

#Filter
grouped_titles = df.groupby("title")
title_count = grouped_titles.size().sort_values(ascending=False)

condition = lambda x: len(x.index) > 1
dup_titles_df = grouped_titles.filter(condition)
dup_titles_df.sort_values("title", inplace=True)

