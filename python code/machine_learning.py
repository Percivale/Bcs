# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 13:58:09 2022

@author: kajah
"""
import pandas as pd
import numpy as np

simple_path = "simple_dataset.xlsx"

simple_df = pd.read_excel(simple_path, index_col=0)
#%%
print(simple_df.head())

#Need to split dataset

#60 percent will be training. 10 will be validation
simple_train = simple_df.sample(frac = 0.6, random_state = 1)
simple_test = simple_df.drop(simple_train.index).sample(frac=0.75, random_state = 1)
simple_valid = simple_df.drop(simple_test.index).drop(simple_train.index)


print("Training set: ", len(simple_train), "\nTesting set: ", len(simple_test), "\nValidation set: ", len(simple_valid))

print(simple_train.head())
#%%




#30 percent will be testing.