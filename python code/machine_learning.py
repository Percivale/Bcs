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
    
def error(pred, true):
    f_pos = np.count_nonzero(pred>true)
    f_neg = np.count_nonzero(pred<true)
    right = np.count_nonzero(np.logical_and(pred !=0, true != 0))
    return f_pos, f_neg, right
    

def feature_importance(model, df):
    importance = model.get_feature_importance()
    col_names = df.columns
    print("\nFeature importance: ")
    for i in range(len(importance)):
        print(col_names[i], ": ", importance[i])
# def find_lr(x_train, y_train, lr_start, lr_end, lr_steps, iterations = 1000, dataset_name = ""):
#     l = np.arange(lr_start, lr_end, lr_steps)
#     plt.figure()
#     plt.title("Error as a function of learning rate" + dataset_name)
#     for k in range(len(l)):
#         l_rate = l[k]
#         weight = y_train*47 + 0.1
#         model_red = CatBoostClassifier(iterations = iterations, learning_rate = l_rate, random_seed = 1) 
#         model_red.fit(Pool(x_train_red, y_train, weight = weight), silent = True)
#         pred_train_w = model_red.predict(x_train)
        
#         if k == 0:
#             f_pos, f_neg, right = error(pred_train_w, y_train)
#             plt.plot(l_rate, f_pos, "go", label = "false positive, training")
#             plt.plot(l_rate, f_neg, "orange", marker = "x", label = "false negative, training")
#             plt.plot(l_rate, right, "purple",marker = "d", label = "#right defects, training")
#         else:

#             f_pos, f_neg, right = error(pred_train_w, y_train)
#             plt.plot(l_rate, f_pos, "go")
#             plt.plot(l_rate, f_neg, "orange",marker = "x")
#             plt.plot(l_rate, right, "purple",marker = "d")
            
#     plt.legend(bbox_to_anchor=(1, 0.5))
#     plt.xlabel("nr of trees")
#     plt.ylabel("Nr of Si atoms")
#     plt.show()

#%%
final, defect_idx = ao.analyze_Si("C:\\Users\\kajah\\git_repo\\Bcs\\xyz_files")

rings, defect_idx = ao.analyze_rings("C:\\Users\\kajah\\git_repo\\Bcs\\xyz_files")

rings_3 = rings[:, 1].reshape((len(rings), 1))

defect_idx = np.array(defect_idx)


post_idx = defect_idx
def_idx = defect_idx+72

target = np.zeros((len(final), 1))
target[def_idx] = 1
final = np.concatenate((final, rings_3), axis = 1)
dataset = pd.DataFrame(np.concatenate((final, target), axis = 1)).reset_index(drop = True)

post_box = dataset[dataset.columns[0]].iloc[post_idx]
post_box = np.array(post_box)

for i in range(len(post_box)):
    dataset = dataset.drop(dataset.loc[dataset[dataset.columns[0]]==post_box[i]].index)
#%%
print(dataset.columns)
print(dataset.head)
dataset.columns = ["Box id", "Silicon id", "SiOSi angle 1", "SiOSi angle 2", "SiOSi angle 3", "SiOSi angle 4",
                   "OSiO angle 1", "OSiO angle 2", "OSiO angle 3", "OSiO angle 4", "OSiO angle 5", "OSiO angle 6",
                   "Bond 1", "Bond 2", "Bond 3", "Bond 4", "Rings of size 3", "Defect"]
print(dataset.head)
print(dataset.iloc[0])
#%%
w_defect = 0.87
w_normal = 1

weights = pd.Series(data = np.full(len(dataset.index), w_normal), index = dataset.index)
weights[def_idx] = w_defect


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
new_data = pd.read_excel("C:\\Users\\kajah\\git_repo\\Bcs\\datasets\\no_post.xlsx")
new_y = new_data["Defect"]
new_data.drop(columns = ["Defect", new_data.columns[0], new_data.columns[1], new_data.columns[2]], inplace = True)
print(new_data.columns)
#%%



#Try another dataset with cut_off angles and more
x_train_temp = x_train[x_train.columns[4:10]]
x_valid_temp = x_valid[x_valid.columns[4:10]]
x_test_temp = x_test[x_test.columns[4:10]]

x_train_new = pd.DataFrame()
#x_train_new["max Si-O-Si"] = x_train[x_train.columns[2]]
#x_train_new["min Si-O-Si"] = x_train[x_train.columns[5]]
#x_train_new["max O-Si-O"] =  x_train[x_train.columns[6]]
#x_train_new["min O-Si-O"] = x_train[x_train.columns[11]]
#x_train_new["O-Si-O smaller than 115"] = np.count_nonzero(x_train_temp<115, axis = 1)
#x_train_new["rings 3"] = x_train[x_train.columns[-1]]
#x_train_new = pd.concat([x_train_new, x_train_temp], axis = 1)
x_train_new = pd.concat([x_train_new, x_train], axis = 1)

x_valid_new = pd.DataFrame()
#x_valid_new["max Si-O-Si"] = x_valid[x_valid.columns[2]]
#x_valid_new["min Si-O-Si"] = x_valid[x_valid.columns[5]]
#x_valid_new["max O-Si-O"] =  x_valid[x_valid.columns[6]]
#x_valid_new["min O-Si-O"] = x_valid[x_valid.columns[11]]
#x_valid_new["O-Si-O smaller than 115"] = np.count_nonzero(x_valid_temp<115, axis = 1)
#x_valid_new["rings 3"] = x_valid[x_valid.columns[-1]]
#x_valid_new = pd.concat([x_valid_new, x_valid_temp], axis = 1)
x_valid_new = pd.concat([x_valid_new, x_valid], axis = 1)
print(x_train_new.shape)

x_test_new = pd.DataFrame()
x_test_new = pd.concat([x_test_new, x_test], axis = 1)

# cut_offs = [125, 130, 132]
# for angle in cut_offs:
#     x_train_new["O-Si-O larger than "+str(angle)] = np.count_nonzero(x_train_temp>angle, axis = 1)
#     x_valid_new["O-Si-O larger than " + str(angle)] = np.count_nonzero(x_valid_temp>angle, axis = 1)
#     x_test_new["O-Si-O larger than " + str(angle)] = np.count_nonzero(x_test_temp>angle, axis = 1)
# print(x_train_new.shape)
# print(np.count_nonzero(np.count_nonzero(x_train_temp>132, axis = 1)))
#%%
model = CatBoostClassifier(random_seed = 1) #Good starting point for validation as well. 77 defects guessed in training
#weight = y_train*105 + 0.1
weight = y_train*41 + 0.1

model.fit(Pool(x_train_new, y_train, weight = weight), silent = True)

pred_train = model.predict(x_train_new)
pred_val = model.predict(x_valid_new)
pred_test = model.predict(x_test_new)
pred_new = model.predict(new_data)

print("\nTraining, weighted")
get_res(pred_train, y_train) 

print("\nValidation")
get_res(pred_val, y_valid)

# print("\nTest:")
# get_res(pred_test, y_test)

print("\n New data")
get_res(pred_new, new_y)

feature_importance(model, x_train_new)
#%%
model.get_all_params()

#%%
model = CatBoostClassifier(random_seed = 1, depth = 8, iterations = 1200, learning_rate = 0.03) #Good starting point for validation as well. 77 defects guessed in training
#weight = y_train*105 + 0.1
weight = y_train*41 + 0.1

model.fit(Pool(x_train_new, y_train, weight = weight), silent = True)

pred_train = model.predict(x_train_new)
pred_val = model.predict(x_valid_new)

print("\nTraining, weighted")
get_res(pred_train, y_train) 

print("\nValidation")
get_res(pred_val, y_valid)
#%%
model = CatBoostClassifier(random_seed = 1, depth = 8, iterations = 1200, learning_rate = 0.027) #Good starting point for validation as well. 77 defects guessed in training
#weight = y_train*105 + 0.1
weight = y_train*41 + 0.1

model.fit(Pool(x_train_new, y_train, weight = weight), silent = True)

pred_train = model.predict(x_train_new)
pred_val = model.predict(x_valid_new)

print("\nTraining, weighted")
get_res(pred_train, y_train) 

print("\nValidation")
get_res(pred_val, y_valid)

#%%
model_rf= RandomForestClassifier(random_state = 1, n_estimators = 2000, max_depth = 4)
weight = y_train*2 + 0.1

model_rf.fit(x_train_new, y_train, sample_weight=weight)

pred_train = model_rf.predict(x_train_new)
pred_val = model_rf.predict(x_valid_new)
print("\nTraining, weighted")
get_res(pred_train, y_train) 

print("\nValidation")
get_res(pred_val, y_valid)
#%%
print(model_rf.get_params())

#%%    

model = CatBoostClassifier(random_seed = 1, learning_rate = 0.015, iterations = 2000, max_depth = 8)
weight = y_train*47 + 0.1
model.fit(Pool(x_train_new, y_train, weight = weight), silent = True)

pred_train = model.predict(x_train_new)
print("\nTraining, weighted")
get_res(pred_train, y_train) 

pred_val = model.predict(x_valid_new)
pred_test = model.predict(x_test_new)

print("\nValidation")
get_res(pred_val, y_valid)

print("\nTest:")
get_res(pred_test, y_test)


"""
Training, weighted
Results: 
Number of defects predicted:  37
True number of defects:  34
Percentage of correct guesses:  99.97856683575051
Percentage of correct defects:  100.0

Validation
Results: 
Number of defects predicted:  8
True number of defects:  8
Percentage of correct guesses:  100.0
Percentage of correct defects:  100.0

Test:
Results: 
Number of defects predicted:  18
True number of defects:  16
Percentage of correct guesses:  99.97142040583023
Percentage of correct defects:  100.0
"""


#%%
feature_importance(model, x_train_new)
#%%
#making a new model predicting on the predictions
x_train2 = x_train_new.loc[pred_train.astype(bool)]
y_train2 = y_train.loc[pred_train.astype(bool)]


model2 = CatBoostClassifier(random_seed = 1, silent = True)
model2.fit(x_train2, y_train2)

pred2 = model2.predict(x_train2)
pred_val = model.predict(x_valid_new)
pred_test = model.predict(x_test_new)

print("Training")
get_res(pred2, y_train2)

print("\nValidation")
get_res(pred_val, y_valid)

print("\nTest")
get_res(pred_test, y_test)

#%%
feature_importance(model2, x_train_new)

#%%
#Guessed 66
model = CatBoostClassifier(random_seed = 1, learning_rate = 0.005, iterations = 2000, max_depth = 12)
weight = y_train*47 + 0.1
model.fit(Pool(x_train_new, y_train, weight = weight), silent = True)

pred_train = model.predict(x_train_new)
pred_val = model.predict(x_valid_new)
print("\nTraining, weighted")
get_res(pred_train, y_train)
print("\nValidation")
get_res(pred_val, y_valid)
#%%
model = CatBoostClassifier(random_seed = 1, learning_rate = 0.015, iterations = 2000, max_depth = 12)
weight = y_train*47 + 0.1
model.fit(Pool(x_train_new, y_train, weight = weight), silent = True)

pred_train = model.predict(x_train_new)
pred_val = model.predict(x_valid_new)
print("\nTraining, weighted")
get_res(pred_train, y_train)
print("\nValidation")
get_res(pred_val, y_valid)
#%%
#Make new dataset with cutoff angles
x_train_temp = x_train[x_train.columns[6:12]]
x_train_ex = x_train
x_train_ex["cut_off 132"] = np.count_nonzero(x_train_temp>132, axis = 1)
x_valid_temp = x_valid[x_valid.columns[6:12]]
x_valid_ex = x_valid
x_valid_ex["cut_off 132"] = np.count_nonzero(x_valid_temp>132, axis = 1)

x_train_co = pd.DataFrame()
x_valid_co = pd.DataFrame()

cut_offs = np.arange(123, 133, 1)
for angle in cut_offs:
    x_train_co["cut-off "+str(angle)] = np.count_nonzero(x_train_temp>angle, axis = 1)
    x_valid_co["cut-off " + str(angle)] = np.count_nonzero(x_valid_temp>angle, axis = 1)
#%%
model = CatBoostClassifier(random_seed = 1)
weight = y_train*3 + 0.1

model.fit(Pool(x_train_co, y_train, weight = weight), silent = True)

pred_train = model.predict(x_train_co)
pred_val = model.predict(x_valid_co)

model.fit(x_train_co, y_train, silent = True)
pred_nw = model.predict(x_train_co)


print("\nTraining, weighted")
get_res(pred_train, y_train) 
print("\nTraining, not weighted")
get_res(pred_nw, y_train)
#%%
get_res(pred_val, y_valid)
#%%
model = CatBoostClassifier(random_seed = 1, learning_rate = 0.025, iterations = 1500, depth = 8)
weight = y_train*3 + 0.1

model.fit(Pool(x_train_co, y_train, weight = weight), silent = True)

pred_train = model.predict(x_train_co)

print("\nTraining, weighted")
get_res(pred_train, y_train) 
#%%
print(model.get_all_params())
#%%
model = CatBoostClassifier(random_seed = 1)
weight = y_train*3 + 0.1

model.fit(Pool(x_train_ex, y_train, weight = weight), silent = True)

pred_train = model.predict(x_train_ex)
pred_val = model.predict(x_valid_ex)

print("\nTraining, weighted")
get_res(pred_train, y_train) 
get_res(pred_val, y_valid)
#%%
print(model.get_all_params())
print(model.get_feature_importance())
#%%
#This improved results. Guessed 47 defects, there are 35 (so six less guesses) Increased the max depth to 8
model = CatBoostClassifier(random_seed = 1, iterations = 1000, learning_rate = 0.027, max_depth=8)
weight = y_train*3 + 0.1

model.fit(Pool(x_train_ex, y_train, weight = weight), silent = True)

pred_train = model.predict(x_train_ex)

print("\nTraining, weighted")
get_res(pred_train, y_train) 
#%%
print(model.get_all_params())
print(model.get_feature_importance())
#%%
#This improved results. Guessed 45 defects (to less wrong). Increased iterations by 200. I might be overfitting...
model = CatBoostClassifier(random_seed = 1, iterations = 1200, learning_rate = 0.027, max_depth=8)
weight = y_train*3 + 0.1

model.fit(Pool(x_train_ex, y_train, weight = weight), silent = True)

pred_train = model.predict(x_train_ex)

print("\nTraining, weighted")
get_res(pred_train, y_train) 
#%%
print(model.get_all_params())
#%%
#This improved results. Guessed 43 defects(two less wrong). The learning rate was 0.028853999450802803
model = CatBoostClassifier(random_seed = 1, iterations = 1200, max_depth=8)
weight = y_train*47 + 0.1

model.fit(Pool(x_train_ex, y_train, weight = weight), silent = True)

pred_train = model.predict(x_train_ex)
pred_val = model.predict(x_valid_ex)
print("\nTraining, weighted")
get_res(pred_train, y_train) 

get_res(pred_val, y_valid)
#%%
print(model.get_all_params())
#%%
#Let's try removing some columns..

x_train_red = x_train_ex[x_train_ex.columns[[0, 5, 6, 11, 12, 19, -1]]]
print(x_train_red.iloc[0])
#%%
#New record! Guessed 42 defects (one less wrong). Increased max depth and set a learning rate.
model = CatBoostClassifier(random_seed = 1, learning_rate = 0.038, iterations = 900, max_depth = 10)
weight = y_train*3 + 0.1

model.fit(Pool(x_train_red, y_train, weight = weight), silent = True)

pred_train = model.predict(x_train_red)

print("\nTraining, weighted")
get_res(pred_train, y_train) 

#%%
print(model.get_all_params())
print(model.get_feature_importance())
#%%
model = CatBoostClassifier(random_seed = 1, iterations = 1700, max_depth = 10)
weight = y_train*3 + 0.1

model.fit(Pool(x_train_red, y_train, weight = weight), silent = True)

pred_train = model.predict(x_train_red)

print("\nTraining, weighted")
get_res(pred_train, y_train) 
print("Validation set")
pred_val = model.predict(x_valid_ex)
get_res(pred_val, y_valid)
#%%
print(model.get_all_params())

#%%
#fitting model
weight = y_train*47 + 0.1
model = CatBoostClassifier(random_seed = 1)
model.fit(Pool(x_train, y_train, weight = weight))

#predicting
prediction = model.predict(x_valid)
pred_train = model.predict(x_train)
#pred_prob= model.predict_proba(x_test) #get probabilities
pred_test = model.predict(x_test)


#Fitting without weights
model.fit(x_train, y_train)

#predicting

prediction_2 = model.predict(x_valid)
pred_train_2 = model.predict(x_train)

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
print("\ntest")
get_res(pred_test, y_test)

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
#find_lr(x_train, y_train, lr_start = 0.01, lr_end=0.5, lr_steps=0.001, dataset_name = " training")

#%%
print(model_red.get_feature_importance())
print(x_train.iloc[0])
#%%
"""
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
l = np.arange(0.015, 0.035, 0.002)
for k in range(len(l)):
    l_rate = l[k]
    weight = y_train*47 + 0.1
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

"""