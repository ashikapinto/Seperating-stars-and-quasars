
# coding: utf-8

# In[5]:


#import libraries
import pandas as pd
import numpy as np
import operator
from csv import reader
from math import sqrt

#Upsampling
data=pd.read_csv('C:\\Users\\HP\\Desktop\\ML\\catalog3\\cat3.csv')
df_majority=data[data['class']==1]
df_minority = data[data['class']==0]
df_minority_random=df_minority.sample(n=len(df_majority),replace=True)
df_upsampled=pd.concat([df_majority,df_minority_random],axis=0)


x=df_upsampled
#x=df_upsampled.drop('class',axis=1)
x=x.drop('galex_objid',axis=1)
x=x.drop('sdss_objid',axis=1)
#x=x.drop('spectrometric_redshift',axis=1)
x=x.drop('pred',axis=1)
x=x.drop('Unnamed: 0',axis=1)
#converting into np.array
x=x.values


dataset=x



# In[6]:


def mean(df):
    means_of_cols = [0 for i in range(len(df[0]))]
    for i in range(len(df[0])):
        column = [row[i] for row in df]
        means_of_cols[i] = sum(column) / float(len(df))
    return means_of_cols

#Function to calculate the standard deviations of each of the columns
def standard_deviation(df, means_of_cols):
    std_of_cols = [0 for i in range(len(df[0]))]
    for i in range(len(df[0])):
        variance = [pow(x[i]-means_of_cols[i], 2) for x in df]
        std_of_cols[i] = sum(variance)
    std_of_cols = [sqrt(x/(float(len(df)-1))) for x in std_of_cols]
    return std_of_cols

# standardize
def standardize(df, means_of_cols, std_of_cols):
    for row in df:
        for i in range(len(row)):
            row[i] = (row[i] - means_of_cols[i]) / std_of_cols[i]


# function to calculate euclidean distance
def distance(x1, x2, n):
    d = 0
    for x in range(n):
        d += np.square(x1[x] - x2[x])
    return np.sqrt(d)

# KNN model
def knn(train, y_train,test_row, k):
    dist = {}
    n = test_row.shape[0]
    for x in range(len(train)):
        d = distance(test_row, train[x], n)
        dist[x] = d
    #Ordered_dist contains the row numbers of the rows of the rows of the training data set ordered in the ascending order of 
    #the distance to the test instance
    ordered_dist = sorted(dist.items(), key=operator.itemgetter(1))
    neighbors = []
    for i in range(k):
        neighbors.append(ordered_dist[i][0])
    #Neighbors contains only the first K elements of the Ordered_dist
    counts = {}  
    for i in range(len(neighbors)):
        target = y_train[neighbors[i]]
        if target in counts:
            counts[target] += 1
        else:
            counts[target] = 1
    #Counts is a dictionary that keeps track of how many votes each of the outputs have recieved 
    ordered_count = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
    #Ordered count sorts this dictionary based on the number of votes and the function returns the target class with the maximum
    #number of votes
    return (ordered_count[0][0]) 

def accuracy(y_test, predictions):
    correct = 0
    for x in range(len(y_test)):
        if y_test[x] == predictions[x]:
            correct += 1
    return (correct/float(len(y_test))) * 100.0



# In[7]:


from random import randrange

def split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)

    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            to_append = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(to_append))
        dataset_split.append(fold)
    return dataset_split

def kfold(dataset, n_folds):
    folds = split(dataset, n_folds)
    #print(folds[0])
    acc=[]
    for i in range(len(folds)):
        train_set = list(folds)
        train_set.pop(i)
        train_set = sum(train_set, [])
        test_set=folds[i]
        #print(len(train_set))
        #print(len(test_set))
        y_train=[]
        y_test=[]
        #index 12 has the class
        for i in train_set:
            y_train.append(list(i).pop(12))
        for i in test_set:
            y_test.append(list(i).pop(12))
        X_train=train_set
        X_test=test_set
        
        means_of_cols = mean(X_train)
        std_of_cols = standard_deviation(X_train, means_of_cols)


        standardize(X_train, means_of_cols, std_of_cols)
        standardize(X_test, means_of_cols, std_of_cols)
        
        predictions=[]
        for i in range(len(X_test)):
            predictions.append(knn(X_train,y_train,X_test[i],3))
        print(accuracy(y_test,predictions))
        acc.append(accuracy(y_test,predictions))
    print("average 10 fold accuracy: ",sum(acc)/len(acc))   
            
        
kfold(dataset,10)

