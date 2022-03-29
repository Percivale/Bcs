# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 10:49:25 2022

@author: kajah
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import AmorphousOxide_ as ao

np.random.seed(123)

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
weight = y_train*10 + 0.1

x_valid = valid.drop(columns = train.columns[-1]).reset_index(drop = True)
y_valid = valid[valid.columns[-1]].reset_index(drop = True)

x_train = x_train.drop(columns = x_train.columns[[0,1]])
x_valid = x_valid.drop(columns = x_valid.columns[[0,1]])

print(max(weight)/min(weight))
print(x_train.shape)
print(y_train.shape)

x_test = test.drop(columns = test.columns[[-1, 0, 1]]).reset_index(drop = True)
#%%

num_classes = 2 # 0=normal Si, 1=defect

# number of training rounds to do
epochs = 800

modelName = 'defecter-detector-v1'


######## DIRECT TRAINING ############
# this will need some adjusting
######## MODEL 1.0
model = Sequential()
model.add(layers.Conv2D(4, (3, 3), strides=2, activation='relu',
input_shape=(None,None,1), data_format="channels_last"))
model.add(layers.Conv2D(8, (3, 3), strides=2, activation='relu',
data_format="channels_last"))
model.add(layers.Conv2D(16, (3, 3), strides=2, activation='relu',
data_format="channels_last"))

model.add(layers.GlobalAveragePooling2D(data_format="channels_last",
keepdims=False))

model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(num_classes, activation='softmax'))


################## serialize model to JSON
#model_json = model.to_json()
#with open(modelName + ".json", "w") as json_file:
#    json_file.write(model_json)
#model.save(modelName + '.h5')

# load parameters
if "-load" in sys.argv:
        model.load_weights(modelName + ".best.hdf5")
        print("weights loaded")



################ NEW MODEL CREATION
opt = keras.optimizers.Adam()
model.compile(loss=keras.losses.categorical_crossentropy,
optimizer=opt, metrics=['accuracy'])

model_json = model.to_json()
with open(modelName + ".json", "w") as json_file:
        json_file.write(model_json)
model.save_weights(modelName + '_Weights.h5')
model.save(modelName + '.h5')

# checkpoint
#filepath= modelName + ".best.hdf5"
filepath = "./models/" + modelName + "."+"{epoch:02d}.hdf5" #
{val_loss:.4f}-{val_accuracy:.4f}
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy',
verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]


# this training command also needs some adjustment!
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
verbose=1, callbacks=callbacks_list, validation_data=(x_test, y_test))

# save after training is done
model.save_weights(modelName + '_Weights.h5')

################## serialize model to JSON

model_json = model.to_json()
with open(modelName + ".json", "w") as json_file:
        json_file.write(model_json)
model.save_weights(modelName + '_Weights.h5')
model.save(modelName + '.h5')
