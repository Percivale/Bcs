# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:56:23 2022

@author: kajah
"""
import numpy as np
import pandas as pd
import AmorphousOxide_ as ao
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from catboost import Pool
import seaborn as sns
"""
final, defect_idx = ao.analyze_Si("C:\\Users\\kajah\\git_repo\\Bcs\\xyz_files_new\\xyz", defect_known=True)

rings, defect_idx = ao.analyze_rings("C:\\Users\\kajah\\git_repo\\Bcs\\xyz_files_new\\xyz", defect_known=True)

dihedral_angles, dihedral_idx, defect_idx = ao.analyze_diheds("C:\\Users\\kajah\\git_repo\\Bcs\\xyz_files_new\\xyz", defect_known=True)

rings_3 = rings[:, 1].reshape((len(rings), 1))

defect_idx = ao.find_defect("C:\\Users\\kajah\\git_repo\\Bcs\\xyz_files_new\\out")
#%%

post_idx = defect_idx
def_idx = defect_idx+72

target = np.zeros((len(final), 1))
target[def_idx] = 1
target[post_idx] = 1
final = np.concatenate((final, rings_3), axis = 1)
#%%

dataset = pd.DataFrame(np.concatenate((final, target), axis = 1)).reset_index(drop = True)
dataset.columns = ["Box id", "Silicon id", "SiOSi angle 1", "SiOSi angle 2", "SiOSi angle 3", "SiOSi angle 4",
                   "OSiO angle 1", "OSiO angle 2", "OSiO angle 3", "OSiO angle 4", "OSiO angle 5", "OSiO angle 6",
                   "Bond 1", "Bond 2", "Bond 3", "Bond 4", "Rings of size 3", "Defect"]

rings_data = pd.DataFrame(np.concatenate((rings, target), axis = 1)).reset_index(drop = True)
rings_data.columns = ["Rings of size 2", "Rings of size 3", "Rings of size 4", "Rings of size 5", "Defect"]

dihed_data = pd.DataFrame(np.concatenate((dihedral_angles, target), axis = 1)).reset_index(drop = True)

post_box = dataset[dataset.columns[0]].iloc[post_idx]
post_box = np.array(post_box)

post_set = pd.DataFrame()
post_rings = pd.DataFrame()
post_dihed = pd.DataFrame()

for i in range(len(post_box)):
    post_set = pd.concat([post_set, dataset.loc[dataset[dataset.columns[0]] == post_box[i]]])
    post_rings = pd.concat([post_rings, rings_data.loc[dataset.loc[dataset[dataset.columns[0]] == post_box[i]].index]])
    post_dihed = pd.concat([post_dihed, dihed_data.loc[dataset.loc[dataset[dataset.columns[0]] == post_box[i]].index]])
    
    dihed_data = dihed_data.drop(dataset.loc[dataset[dataset.columns[0]]==post_box[i]].index)
    rings_data = rings_data.drop(dataset.loc[dataset[dataset.columns[0]]==post_box[i]].index)
    dataset = dataset.drop(dataset.loc[dataset[dataset.columns[0]]==post_box[i]].index)
    
  
print(post_rings.head)
print(post_set.head)
print(rings_data.head)
print(dataset.head)


#dataset.to_excel("C:\\Users\\kajah\\git_repo\\Bcs\\datasets\\no_post.xlsx")
#post_set.to_excel("C:\\Users\\kajah\\git_repo\\Bcs\\datasets\\post.xlsx")
#rings_data.to_excel("C:\\Users\\kajah\\git_repo\\Bcs\\datasets\\no_post_rings.xlsx")
#post_rings.to_excel("C:\\Users\\kajah\\git_repo\\Bcs\\datasets\\post_rings.xlsx")
dihed_data.to_excel("C:\\Users\\kajah\\git_repo\\Bcs\\datasets\\dihed.xlsx")
post_dihed.to_excel("C:\\Users\\kajah\\git_repo\\Bcs\\datasets\\post_dihed.xlsx")
"""

no_post = pd.read_excel("C:\\Users\\kajah\\git_repo\\Bcs\\datasets\\no_post.xlsx")
post = pd.read_excel("C:\\Users\\kajah\\git_repo\\Bcs\\datasets\\post.xlsx")
rings = pd.read_excel("C:\\Users\\kajah\\git_repo\\Bcs\\datasets\\no_post_rings.xlsx")
post_rings = pd.read_excel("C:\\Users\\kajah\\git_repo\\Bcs\\datasets\\post_rings.xlsx")
dihed = pd.read_excel("C:\\Users\\kajah\\git_repo\\Bcs\\datasets\\dihed.xlsx")
post_dihed = pd.read_excel("C:\\Users\\kajah\\git_repo\\Bcs\\datasets\\post_dihed.xlsx")

def_pre = no_post[no_post["Defect"] == 1]
def_post = post[post["Defect"] == 1]
pre = no_post[no_post["Defect"] == 0]
post = post[post["Defect"] == 0]

def_rings = rings[rings["Defect"] == 1]
pre_rings = rings[rings["Defect"] == 0]
def_post_rings = post_rings[post_rings["Defect"] == 1]
post_rings = post_rings[post_rings["Defect"] == 0]

def_dihed = dihed[dihed[dihed.columns[-1]] == 1]
dihed = dihed[dihed[dihed.columns[-1]] == 0]
def_dihed_post = post_dihed[post_dihed[post_dihed.columns[-1]] == 1]
dihed_post = post_dihed[post_dihed[post_dihed.columns[-1]] == 0]



def histogram(objs, title, c = ["green", "purple"], label = ["pre", "post"], axis_label = False):
    plt.figure()
    plt.title(title)
    it = 0
    for obj in objs:
        plt.hist(obj, alpha = 0.5, color = c[it], edgecolor = "k", label = label[it], density = True)
        it+=1
    plt.legend()
    if axis_label:
        plt.xlabel(axis_label)
    plt.show
    
#%%
plot_col = def_pre.columns[3:-1]
for i in range(len(plot_col)):
    histogram([def_pre[plot_col[i]], def_post[plot_col[i]]] ,plot_col[i] + " defect sites")

temp = post["Box id"]+1
corr_pre = pd.DataFrame()
for box in temp.unique():
    corr_pre = pd.concat([corr_pre, pre[pre["Box id"] == box]])

for i in range(len(plot_col)):
    histogram([corr_pre[plot_col[i]], post[plot_col[i]]], title = plot_col[i] + " normal sites in defect boxes")

for i in range(len(plot_col)):
    histogram([pre[plot_col[i]], def_pre[plot_col[i]]], plot_col[i] + " defect vs normal sites", c = ["green", "red"], label = ["Normal site", "Defect site"])

#%%
def_osio_pre = def_pre[plot_col[4:10]].to_numpy().flatten()
def_osio_post = def_post[plot_col[4:10]].to_numpy().flatten()
osio_pre = pre[plot_col[4:10]].to_numpy().flatten()
osio_post = post[plot_col[4:10]].to_numpy().flatten()
osio_corr_pre = corr_pre[plot_col[4:10]].to_numpy().flatten()

def_siosi_pre = def_pre[plot_col[:4]].to_numpy().flatten()
def_siosi_post = def_post[plot_col[:4]].to_numpy().flatten()
siosi_post = post[plot_col[:4]].to_numpy().flatten()
siosi_pre = pre[plot_col[:4]].to_numpy().flatten()
siosi_corr_pre = corr_pre[plot_col[:4]].to_numpy().flatten()

def_bond_pre = def_pre[plot_col[10:14]].to_numpy().flatten()
def_bond_post = def_post[plot_col[10:14]].to_numpy().flatten()
bond_post = post[plot_col[10:14]].to_numpy().flatten()
bond_pre = pre[plot_col[10:14]].to_numpy().flatten()
#%% 
histogram([def_osio_pre, def_osio_post], "O-Si-O angles around defects sites", axis_label = "Degrees")
histogram([osio_pre], title = "O-Si-O angles around normal sites", axis_label = "Degrees")
histogram([osio_corr_pre, osio_post], "O-Si-O angles around normal sites in defect boxes", axis_label = "Degrees")
histogram([osio_pre, def_osio_pre], "O-Si-O angles in pre files", c= ["green", "red"], label = ["Normal sites", "Defect sites"], axis_label = "Degrees")

histogram([def_siosi_pre, def_siosi_post], "Si-O-Si angles around defects sites", axis_label = "Degrees")
histogram([siosi_pre], title = "Si-O-Si angles around normal sites", axis_label = "Degrees")
histogram([siosi_corr_pre, siosi_post], "Si-O-Si angles around normal sites", axis_label = "Degrees")
histogram([siosi_pre, def_siosi_pre], "Si-O-Si angles in pre files", c= ["green", "red"], label = ["Normal sites", "Defect sites"], axis_label = "Degrees")

histogram([def_bond_pre, def_bond_post], "Bond lengths around defects sites", axis_label ="Bond length [Å]")
histogram([bond_pre, def_bond_pre], "Bond lengths", c= ["green", "red"], label = ["Normal sites", "Defect sites"],axis_label ="Bond length [Å]")
histogram([bond_pre, def_bond_pre, def_bond_post], "Bond lengths", c= ["green", "red", "purple"], label = ["Normal sites", "Defect sites, pre", "Defect sites, post"],axis_label ="Bond length [Å]")

#%%  
plt.figure()
plt.title("Largest osio vs largest bond, defects")
plt.scatter(def_pre[plot_col[4:10]].to_numpy()[:,0], def_pre[plot_col[10:14]].to_numpy()[:,0])
plt.show()

plt.figure()
plt.title("Largest osio vs largest bond, normal")
plt.scatter(pre[plot_col[4:10]].to_numpy()[:,0], pre[plot_col[10:14]].to_numpy()[:,0])
plt.show()

plt.figure()
plt.title("Largest osio vs largest bond, defects")
plt.scatter(def_pre[plot_col[4:10]].to_numpy()[:,0], def_pre[plot_col[10:14]].to_numpy()[:,-1])
plt.show()

plt.figure()
plt.title("Largest osio vs largest bond, normal")
plt.scatter(pre[plot_col[4:10]].to_numpy()[:,0], pre[plot_col[10:14]].to_numpy()[:,-1])
plt.show()

plt.figure()
plt.title("Largest osio vs smallest, defects")
plt.scatter(def_pre[plot_col[4:10]].to_numpy()[:,0], def_pre[plot_col[4:10]].to_numpy()[:,-1])
plt.show()

plt.figure()
plt.title("Largest osio vs smallest, normal")
plt.scatter(pre[plot_col[4:10]].to_numpy()[:,0], pre[plot_col[4:10]].to_numpy()[:,-1])
plt.scatter(def_pre[plot_col[4:10]].to_numpy()[:,0], def_pre[plot_col[4:10]].to_numpy()[:,-1])

plt.show()

plt.figure()
plt.title("Largest osio vs largest bond, defects")
plt.scatter(def_pre[plot_col[4:10]].to_numpy()[:,0]/def_pre[plot_col[4:10]].to_numpy()[:,-1], def_pre[plot_col[4:10]].to_numpy()[:,0]/def_pre[plot_col[4:10]].to_numpy()[:,-1])
plt.show()

plt.figure()
plt.title("proportion between largest and smalles bond + angle")
plt.scatter(pre[plot_col[4:10]].to_numpy()[:,0]/pre[plot_col[4:10]].to_numpy()[:,-1], pre[plot_col[4:10]].to_numpy()[:,0]/pre[plot_col[4:10]].to_numpy()[:,-1])
plt.scatter(def_pre[plot_col[4:10]].to_numpy()[:,0]/def_pre[plot_col[4:10]].to_numpy()[:,-1], def_pre[plot_col[4:10]].to_numpy()[:,0]/def_pre[plot_col[4:10]].to_numpy()[:,-1])
plt.show()

#%%
print(def_rings)
#%%
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_title("Rings of size 3")
ax2.set_title("Rings of size 4")
ax1.scatter([0,1], [0,0], s = [np.count_nonzero(def_rings["Rings of size 3"] == 0)/len(def_pre)*1000, np.count_nonzero(def_rings[["Rings of size 3"]])/len(def_pre)*1000])
ax1.scatter([0,1], [1, 1], s = [np.count_nonzero(pre_rings[["Rings of size 3"]] == 0)/len(pre)*1000, np.count_nonzero(pre_rings[["Rings of size 3"]])/len(pre)*1000])
ax2.scatter([0,1], [0,0], s = [np.count_nonzero(def_rings["Rings of size 4"] == 0)/len(def_pre)*1000, np.count_nonzero(def_rings[["Rings of size 4"]])/len(def_pre)*1000])
ax2.scatter([0,1], [1, 1], s = [np.count_nonzero(pre_rings[["Rings of size 4"]] == 0)/len(pre)*1000, np.count_nonzero(pre_rings[["Rings of size 4"]])/len(pre)*1000])
ax1.set_xticks([0,1])
ax2.set_xticks([0,1])
ax1.set_yticks([0,1], ["Defect site", "Normal site"])
ax2.set_yticks([0,1], ["Defect site", "Normal site"])
ax1.set_ylim(-0.5, 1.5)
ax1.set_xlim(-0.5, 1.5)
ax2.set_ylim(-0.5, 1.5)
ax2.set_xlim(-0.5, 1.5)
plt.show()

plt.figure("Rings around defects sites")
plt.subplot(111)
plt.title("Rings of size 3")
plt.scatter([0,1], [0,0], s = [np.count_nonzero(def_rings["Rings of size 3"] == 0)/len(def_pre)*1000, np.count_nonzero(def_rings[["Rings of size 3"]])/len(def_pre)*1000])
plt.xticks(ticks = [0,1])
plt.yticks([0,1],["Defect site", "Normal site"])
plt.scatter([0,1], [1, 1], s = [np.count_nonzero(pre_rings[["Rings of size 3"]] == 0)/len(pre)*1000, np.count_nonzero(pre_rings[["Rings of size 3"]])/len(pre)*1000])
plt.ylim(-0.5, 1.5)
plt.xlim(-0.5, 1.5)
plt.subplot(211)
plt.title("Rings of size 4")
plt.scatter([0,1], [0,0], s = [np.count_nonzero(def_rings["Rings of size 4"] == 0)/len(def_pre)*1000, np.count_nonzero(def_rings[["Rings of size 4"]])/len(def_pre)*1000])
plt.xticks(ticks = [0,1])
plt.yticks([0,1],["Defect site", "Normal site"])
plt.scatter([0,1], [1, 1], s = [np.count_nonzero(pre_rings[["Rings of size 4"]] == 0)/len(pre)*1000, np.count_nonzero(pre_rings[["Rings of size 4"]])/len(pre)*1000])
plt.ylim(-0.5, 1.5)
plt.xlim(-0.5, 1.5)
plt.show()
#%%
cutoff_angle = 125
cut_offs = np.arange(105, 135, 1)

num = 0

count_defpre = np.count_nonzero(np.count_nonzero(def_pre[plot_col[4:10]].to_numpy()>cutoff_angle, axis = 1)>num)
count_pre = np.count_nonzero(np.count_nonzero(pre[plot_col[4:10]].to_numpy()>cutoff_angle, axis = 1)>num)
count_defpost = np.count_nonzero(np.count_nonzero(def_post[plot_col[4:10]].to_numpy()>cutoff_angle, axis = 1)>num)


plt.figure()
plt.title("Si atoms with at least one angle larger than the cut-off angle", size = 16)
plt.plot(cutoff_angle, count_pre/len(pre)*100, color = "royalblue", marker = "o", markersize = 8, label = "normal")
plt.plot(cutoff_angle, count_defpre/len(def_pre)*100, color = "darkorange", marker = "x", markersize = 8, label = "pre defect")
plt.plot(cutoff_angle, count_defpost/len(def_post)*100, color = "red", marker = "x", markersize = 8, label = "post defect")

for cutoff_angle in cut_offs:
    count_defpre = np.count_nonzero(np.count_nonzero(def_pre[plot_col[4:10]].to_numpy()>cutoff_angle, axis = 1)>num)
    count_pre = np.count_nonzero(np.count_nonzero(pre[plot_col[4:10]].to_numpy()>cutoff_angle, axis = 1)>num)
    count_defpost = np.count_nonzero(np.count_nonzero(def_post[plot_col[4:10]].to_numpy()>cutoff_angle, axis = 1)>num)
    
    print("\n\nCut-off angle: ", cutoff_angle)
    print("Normal: Nr of occurrences: ", count_pre, ". Percentage : ", np.round(count_pre/len(pre)*100, 2))
    print("Pre defect: Nr of occurrences: ", count_defpre, ". Percentage : ", np.round(count_defpre/len(def_pre)*100, 2))
    print("Post defect: Nr of occurrences: ", count_defpost, ". Percentage : ", np.round(count_defpost/len(def_post)*100, 2))
    
    plt.plot(cutoff_angle, count_pre/len(pre)*100, color = "royalblue", marker = "o", markersize = 8)
    plt.plot(cutoff_angle, count_defpre/len(def_pre)*100, color = "darkorange", marker = "x", markersize = 8)
    plt.plot(cutoff_angle, count_defpost/len(def_post)*100, color = "red", marker = "x", markersize = 8)
    
plt.ylabel("Percentage", size = 16)
plt.xlabel("Cut-off angle", size = 16)
plt.xticks(size = 14)
plt.yticks(size = 14)
plt.legend(fontsize = 12)
plt.show()

#%%
print(def_rings)
#%%
size = [2, 3, 4, 5]
#Maybe add ticks so it looks better
for i in range(1, len(size)):
    histogram([def_rings[def_rings.columns[i]], def_post_rings[def_rings.columns[i]]], title = "Rings of size " + str(i+1) + " around defects")
    histogram([pre_rings[def_rings.columns[i]], def_rings[def_rings.columns[i]]], title = "Rings of size " + str(i+1) + " around normal sites and defect sites", c= ["green", "red"], label = ["Normal sites", "Defect sites"], axis_label = "Number of rings by atom")
    
#%%
#87, 170, asio_063
print(def_rings)
print(def_post_rings)
for col in def_rings.columns:
    print(np.where((np.array(def_rings[col]) == np.array(def_post_rings[col])) != True))
    

#%%
print(def_pre)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(def_rings)
    print(def_post_rings)
    
#%%
print(def_dihed.shape)
print(def_dihed.iloc[0])
print(def_dihed.iloc[0][1:13])
print(def_dihed.iloc[0][13:-1])
#%%
def_first_sort = np.sort(def_dihed.to_numpy()[:,1:13], axis = 1)
def_sec_sort = np.sort(def_dihed.to_numpy()[:, 13:-1], axis = 1)

def_first_sort_post = np.sort(def_dihed_post.to_numpy()[:,1:13], axis = 1)
def_sec_sort_post = np.sort(def_dihed_post.to_numpy()[:, 13:-1], axis = 1)

ndef_first_sort = np.sort(dihed.to_numpy()[:,1:13], axis = 1)
ndef_sec_sort = np.sort(dihed.to_numpy()[:, 13:-1], axis = 1)


plt.title("Dihedrals, defect sites, pre")
plt.hist2d(x = def_first_sort.flatten(), y = def_sec_sort.flatten())
plt.xlabel("Dihedral with silicon atom in position 1")
plt.ylabel("Dihedral with silicon atom in position 2")
plt.grid()
plt.show()

plt.title("Dihedrals, defect sites, post")
plt.hist2d(x = def_first_sort_post.flatten(), y = def_sec_sort_post.flatten())
plt.xlabel("Dihedral with silicon atom in position 1")
plt.ylabel("Dihedral with silicon atom in position 2")
plt.grid()
plt.show()

plt.title("Dihedrals, normal sites")
plt.hist2d(x = ndef_first_sort.flatten(), y = ndef_sec_sort.flatten())
plt.xlabel("Dihedral with silicon atom in position 1")
plt.ylabel("Dihedral with silicon atom in position 2")
plt.grid()
plt.show()





plt.title("Dihedrals, defect sites, pre")
sns.kdeplot(x = def_first_sort.flatten(), y = def_sec_sort.flatten(), cmap="Reds", shade=True)
plt.xlabel("Dihedral with silicon atom in position 1")
plt.ylabel("Dihedral with silicon atom in position 2")
plt.grid()
plt.show()

plt.title("Dihedrals, defect sites, post")
sns.kdeplot(x = def_first_sort_post.flatten(), y = def_sec_sort_post.flatten(), cmap="Reds", shade=True)
plt.xlabel("Dihedral with silicon atom in position 1")
plt.ylabel("Dihedral with silicon atom in position 2")
plt.grid()
plt.show()

plt.title("Dihedrals, normal sites")
sns.kdeplot(x = ndef_first_sort.flatten(), y = ndef_sec_sort.flatten(), cmap="Reds", shade=True)
plt.xlabel("Dihedral with silicon atom in position 1")
plt.ylabel("Dihedral with silicon atom in position 2")
plt.grid()
plt.show()
#%%


