# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 13:58:09 2022

@author: kajah
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import AmorphousOxide_ as ao
from catboost import CatBoostClassifier
from catboost import Pool

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
    
def error(pred, true):
    f_pos = np.count_nonzero(pred>true)
    f_neg = np.count_nonzero(pred<true)
    right = np.count_nonzero(np.logical_and(pred !=0, true != 0))
    return f_pos, f_neg, right
    

#%%
final, defect_idx = ao.analyze_Si("C:\\Users\\kajah\\git_repo\\Bcs\\xyz_files")
 #%%
defect_idx = np.array(defect_idx)
print(final[0])
#%%
post_idx = defect_idx
def_idx = defect_idx+72

w_defect = 0.1
w_normal = 0.09

weights = np.full(len(final), w_normal)
weights[def_idx] = w_defect

target = np.zeros((len(final), 1))
target[def_idx] = 1

dataset = pd.DataFrame(np.concatenate((final, target), axis = 1))

#Splitting dataset
train = dataset.sample(frac = 0.6, random_state=1, weights = weights)
test = dataset.drop(train.index).sample(frac = 0.75, random_state = 1)
valid = dataset.drop(test.index).drop(train.index).sample(frac = 1, random_state = 1)

#60% of defects vs the number of defects in the training set
print("60 percent of defects: ", 58*0.6)
print("Number of defects in training set: ", (train[train.columns[-1]].where(train[train.columns[-1]] == 1)).count())
print("\n30 percent of defects: ", 58*0.3)
print("Number of defects in test set: ", (test[test.columns[-1]].where(test[test.columns[-1]] == 1)).count())
print("\n10 percent of defects: ", round(58*0.1, 1))
print("Number of defects in valid set: ", (valid[valid.columns[-1]].where(valid[valid.columns[-1]] == 1)).count())
#%%
#Splitting training set in attributes and target
x_train = train.drop(columns = train.columns[-1]).reset_index(drop = True)
y_train = train[train.columns[-1]].reset_index(drop = True)
weight = y_train*12 + 0.1

x_valid = valid.drop(columns = train.columns[-1]).reset_index(drop = True)
y_valid = valid[valid.columns[-1]].reset_index(drop = True)


print(max(weight)/min(weight))
print(x_train.shape)
print(y_train.shape)

x_test = test.drop(columns = test.columns[-1]).reset_index(drop = True)

#%%
#fitting model
model = CatBoostClassifier(random_seed = 1)
model.fit(Pool(x_train, y_train, weight = weight))

#predicting
prediction = model.predict(x_valid)
pred_train = model.predict(x_train)
#pred_prob= model.predict_proba(x_test) #get probabilities

#Fitting without weights
model.fit(x_train, y_train)

#predicting

prediction_2 = model.predict(x_valid)
pred_train_2 = model.predict(x_train)

#%%
#With weight:
print("With weights\n------------------------------\nValidation:")
get_res(prediction, y_valid)
print("\nTraining")
get_res(pred_train, y_train) 

#Without weight
print("\nWithout weights\n-----------------------------------\nValidation")
get_res(prediction_2, y_valid)
print("\nTraining")
get_res(pred_train_2, y_train)

#%%
x_train_red = x_train.drop(columns = x_train.columns[[0,1]])
x_valid_red = x_valid.drop(columns = x_valid.columns[[0,1]])

model_red = CatBoostClassifier(random_seed = 1)
model_red.fit(x_train_red, y_train)

pred = model_red.predict(x_valid_red)
pred_train = model_red.predict(x_train_red)

model_red.fit(Pool(x_train_red, y_train, weight = weight))
pred_w = model_red.predict(x_valid_red)
pred_train_w = model_red.predict(x_train)

#%%
#Without weight
print("\nWithout weights\n-----------------------------------\nValidation")
get_res(pred, y_valid)
print("\nTraining")
get_res(pred_train, y_train)

#With weights
print("\nWith weights\n------------------------------\nValidation:")
get_res(pred_w, y_valid)
print("\nTraining")
get_res(pred_train_w, y_train) 

x_train = x_train_red
x_valid = x_valid_red
#%%
#Attempting different weights
step = np.arange(0.01, 1, 0.1)

plt.figure()
plt.title("Error as a function of weight")
for i in step:
    weight = y_train*i + 0.1
    model_red = CatBoostClassifier(iterations = 2000, learning_rate= 0.00025, random_seed = 1)
    model_red.fit(x_train_red, y_train, silent = True)
    
    pred = model_red.predict(x_valid_red)
    pred_train = model_red.predict(x_train_red)
    
    
    
    model_red.fit(Pool(x_train_red, y_train, weight = weight), silent = True)
    pred_w = model_red.predict(x_valid_red)
    pred_train_w = model_red.predict(x_train)
    
    if i == step[0]:
        f_pos, f_neg, right = error(pred_w, y_valid)
        plt.plot(max(weight)/min(weight), f_pos, "bo", label = "false positive, validation")
        plt.plot(max(weight)/min(weight), f_neg, "rx", label = "false negative, validation")
        plt.plot(max(weight)/min(weight), right, "kd", label = "#right defects, validation")
        
        f_pos, f_neg, right = error(pred_train_w, y_train)
        plt.plot(max(weight)/min(weight), f_pos, "go", label = "false positive, training")
        plt.plot(max(weight)/min(weight), f_neg, "orange", marker = "x", label = "false negative, training")
        plt.plot(max(weight)/min(weight), right, "purple",marker = "d", label = "#right defects, training")
    else:
        f_pos, f_neg, right = error(pred_w, y_valid)
        plt.plot(max(weight)/min(weight), f_pos, "bo")
        plt.plot(max(weight)/min(weight), f_neg, "rx")
        plt.plot(max(weight)/min(weight), right, "kd")
        
        f_pos, f_neg, right = error(pred_train_w, y_train)
        plt.plot(max(weight)/min(weight), f_pos, "go")
        plt.plot(max(weight)/min(weight), f_neg, "orange",marker = "x")
        plt.plot(max(weight)/min(weight), right, "purple",marker = "d")
    
    #Without weight
    print("\nWithout weights\n-----------------------------------\nValidation")
    get_res(pred, y_valid)
    print("\nTraining")
    get_res(pred_train, y_train)
    
    print("\n\nWeight proportion: ", np.round(max(weight)/min(weight)), 2)
    #With weights
    print("\nWith weights\n------------------------------\nValidation:")
    get_res(pred_w, y_valid)
    print("\nTraining")
    get_res(pred_train_w, y_train) 
plt.legend(bbox_to_anchor=(1, 0.5))
plt.xlabel("weight proportion")
plt.ylabel("Nr of Si atoms")
plt.show()
#%%
l = np.arange(0.0005, 0.0025, 0.0002)
for k in range(len(l)):
    l_rate = l[k]
    weight = y_train*step[2] + 0.1
    plt.figure()
    plt.title("Error as a function of #trees, learning rate = " + str(np.round(l_rate, 4)))
    for i in range(500,2500, 100):
        model_red = CatBoostClassifier(iterations = i, learning_rate = l_rate, random_seed = 1) 
        model_red.fit(Pool(x_train_red, y_train, weight = weight), silent = True)
        #pred_w = model_red.predict(x_valid_red)
        pred_train_w = model_red.predict(x_train)
        
        if i == 500:
            #f_pos, f_neg, right = error(pred_w, y_valid)
            #plt.plot(i, f_pos, "bo", label = "false positive, validation")
            #plt.plot(i, f_neg, "rx", label = "false negative, validation")
            #plt.plot(i, right, "kd", label = "#right defects, validation")
            
            f_pos, f_neg, right = error(pred_train_w, y_train)
            plt.plot(i, f_pos, "go", label = "false positive, training")
            plt.plot(i, f_neg, "orange", marker = "x", label = "false negative, training")
            plt.plot(i, right, "purple",marker = "d", label = "#right defects, training")
        else:
            #f_pos, f_neg, right = error(pred_w, y_valid)
            #plt.plot(i, f_pos, "bo")
            #plt.plot(i, f_neg, "rx")
            #plt.plot(i, right, "kd")
            
            f_pos, f_neg, right = error(pred_train_w, y_train)
            plt.plot(i, f_pos, "go")
            plt.plot(i, f_neg, "orange",marker = "x")
            plt.plot(i, right, "purple",marker = "d")
        
    plt.legend(bbox_to_anchor=(1, 0.5))
    plt.xlabel("nr of trees")
    plt.ylabel("Nr of Si atoms")
    plt.show()

