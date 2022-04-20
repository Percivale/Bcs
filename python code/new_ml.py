# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:56:36 2022

@author: kajah
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import AmorphousOxide_ as ao
from catboost import CatBoostClassifier
from catboost import Pool
from sklearn.ensemble import RandomForestClassifier

np.random.seed(123)

def get_res(pred, true_pred):
    print("Results: ")
    pred_def = np.count_nonzero(pred)
    defects = np.count_nonzero(true_pred)
    print("Number of defects predicted: ", pred_def)
    print("True number of defects: ", defects)
    
    right_idx = np.count_nonzero(np.logical_and(pred !=0, true_pred != 0))
    print("Percentage of correct guesses: ", np.count_nonzero(pred == true_pred)/len(pred)*100)
    print("Percentage of correct defects: ", right_idx/np.count_nonzero(true_pred)*100)

def feature_importance(model, df): #For catboost
    importance = model.get_feature_importance()
    col_names = df.columns
    print("\nFeature importance: ")
    for i in range(len(importance)):
        print(col_names[i], ": ", importance[i])
        

dataset = pd.read_excel("C:\\Users\\kajah\\git_repo\\Bcs\\datasets\\no_post.xlsx")
dataset.drop(columns = dataset.columns[0], inplace = True)
#%%        

w_defect = 0.87
w_normal = 1

weights = pd.Series(data = np.full(len(dataset.index), w_normal), index = dataset.index)
weights[dataset[dataset["Defect"] == 1].index] = w_defect


#Splitting dataset
train = dataset.sample(frac = 0.6, random_state=1, weights = weights)
test = dataset.drop(train.index).sample(frac = 0.75, random_state = 1)
valid = dataset.drop(test.index).drop(train.index).sample(frac = 1, random_state = 1)

#60% of defects vs the number of defects in the training set
print("60 percent of defects: ", 19*0.6)
print("Number of defects in training set: ", (train[train.columns[-1]].where(train[train.columns[-1]] == 1)).count())
print("\n30 percent of defects: ", 19*0.3)
print("Number of defects in test set: ", (test[test.columns[-1]].where(test[test.columns[-1]] == 1)).count())
print("\n10 percent of defects: ", round(19*0.1, 1))
print("Number of defects in valid set: ", (valid[valid.columns[-1]].where(valid[valid.columns[-1]] == 1)).count())#%%
#Splitting training set in attributes and target
x_train = train.drop(columns = train.columns[[0,1,-1]]).reset_index(drop = True)
y_train = train[train.columns[-1]].reset_index(drop = True)
weight = y_train*12 + 0.1

x_valid = valid.drop(columns = train.columns[[0,1,-1]]).reset_index(drop = True)
y_valid = valid[valid.columns[-1]].reset_index(drop = True)


print(max(weight)/min(weight))
print(x_train.shape)
print(y_train.shape)
print(x_train.loc[0])
x_test = test.drop(columns = test.columns[[0,1,-1]]).reset_index(drop = True)
y_test = test[test.columns[-1]].reset_index(drop = True)
#%%
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(x_valid.loc[y_valid == 1])
#%%
model = CatBoostClassifier(random_seed = 1) #Good starting point for validation as well. 77 defects guessed in training
#weight = y_train*105 + 0.1
weight = y_train*46 + 0.1

model.fit(Pool(x_train, y_train, weight = weight), silent = True)

pred_train = model.predict(x_train)
pred_val = model.predict(x_valid)

print("\nTraining, weighted")
get_res(pred_train, y_train) 

print("\nValidation")
get_res(pred_val, y_valid)

feature_importance(model, x_train)

#%%
x_train2 = x_train[["SiOSi angle 1", "SiOSi angle 4", "OSiO angle 1", "OSiO angle 6", "Bond 1", "Bond 4"]]

model = CatBoostClassifier(random_seed = 1) #Good starting point for validation as well. 77 defects guessed in training
#weight = y_train*105 + 0.1
weight = y_train*46 + 0.1

model.fit(Pool(x_train2, y_train, weight = weight), silent = True)

pred_train = model.predict(x_train2)
pred_val = model.predict(x_valid)

print("\nTraining, weighted")
get_res(pred_train, y_train) 

print("\nValidation")
get_res(pred_val, y_valid)

feature_importance(model, x_train2)
#%%
x_train2 = x_train[["SiOSi angle 1", "SiOSi angle 4", "OSiO angle 1", "OSiO angle 6", "Bond 1", "Bond 2"]]

model = CatBoostClassifier(random_seed = 1) #Good starting point for validation as well. 77 defects guessed in training
#weight = y_train*105 + 0.1
weight = y_train*46 + 0.1

model.fit(Pool(x_train2, y_train, weight = weight), silent = True)

pred_train = model.predict(x_train2)
pred_val = model.predict(x_valid)

print("\nTraining, weighted")
get_res(pred_train, y_train) 

print("\nValidation")
get_res(pred_val, y_valid)

feature_importance(model, x_train2)

#%%
x_train2 = x_train[["SiOSi angle 1", "SiOSi angle 4", "OSiO angle 1", "OSiO angle 6", "Bond 1", "Bond 4"]]

x_train2["delta siosi"] = x_train["SiOSi angle 1"] - x_train["SiOSi angle 4"]
#%%

model = CatBoostClassifier(random_seed = 1) #Good starting point for validation as well. 77 defects guessed in training
#weight = y_train*105 + 0.1
weight = y_train*46 + 0.1

model.fit(Pool(x_train2, y_train, weight = weight), silent = True)

pred_train = model.predict(x_train2)
pred_val = model.predict(x_valid)

print("\nTraining, weighted")
get_res(pred_train, y_train) 

print("\nValidation")
get_res(pred_val, y_valid)

feature_importance(model, x_train2)