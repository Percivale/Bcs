# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 16:04:54 2022

@author: kajah
"""
#%%
import AmorphousOxide_ as ao
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import pandas as pd


#%%

for subdir, dirs, files in os.walk("C:\\Users\\kajah\\git_repo\\Bcs\\xyz_files_new\\xyz"):
    i = 0
    for file in files:
        print(i, file)
        i+=1


#%%


xyz_path = "C:\\Users\\kajah\\git_repo\\Bcs\\xyz_files\\asio2_015_post.xyz"

cutoffs = [3.44, 2.0, 3.159]
etrap = ao.AmorphousOxide(216, 72, 144, xyz_path)

index, bond_lengths, dr = ao.make_bonds(etrap.xArray, etrap.yArray, etrap.zArray, etrap.nameArray, 
                                        etrap.boxX, etrap.boxY, etrap.boxZ, cutoffs)

##########################################################################
#Bond angles and dihedral angles
sisisi_idx, siosi_idx, osio_idx = ao.match_bonds(index)

#%%
rings, defect_idx = ao.analyze_rings("C:\\Users\\kajah\\git_repo\\Bcs\\xyz_files")

#%%
print(rings.shape, len(defect_idx))
defect_idx = np.array(defect_idx)
#%%
size = [2, 3, 4, 5]

post_rings = rings[defect_idx]
def_rings = rings[defect_idx + 72]
ndef_rings = np.delete(rings, np.append(defect_idx, defect_idx + 71, axis = 0), axis = 0)
plt.figure()
plt.title("Percentage of rings around Si atoms")
plt.plot(size, np.count_nonzero(def_rings, axis = 0)/len(def_rings)*100, "-o", label = "Defect site")
#plt.plot(size, np.count_nonzero(post_rings, axis = 0)/len(post_rings)*100, "-o", label = "Post")
plt.plot(size, np.count_nonzero(ndef_rings, axis = 0)/len(ndef_rings)*100, "-x", label = "Normal site")
plt.xlabel("Number of atoms in ring")
plt.ylabel("Percentage")
plt.legend()
plt.show()

perc_ndef = np.count_nonzero(ndef_rings, axis = 0)/len(ndef_rings)*100
perc_def = np.count_nonzero(def_rings, axis = 0)/len(def_rings)*100
print("Normal sites: \n----------------------")
print( "Rings of size 2: ", perc_ndef[0])
print( "Rings of size 3: ", perc_ndef[1])
print( "Rings of size 4: ", perc_ndef[2])
print( "Rings of size 5: ", perc_ndef[3])
print("\n")

print("Defect sites: \n----------------------")
print( "Rings of size 2: ", perc_def[0])
print( "Rings of size 3: ", perc_def[1])
print( "Rings of size 4: ", perc_def[2])
print( "Rings of size 5: ", perc_ndef[3])
print("\n\n")

perc_ndef = np.count_nonzero(np.logical_and(ndef_rings[:,1] >0, ndef_rings[:, 2]>0))/len(ndef_rings)*100
perc_def = np.count_nonzero(np.logical_and(def_rings[:,1] >0, def_rings[:, 2]>0))/len(def_rings)*100

print("Normal sites: \n----------------------")
print(  "Si in both ring of size 3 and 4", perc_ndef)
print("\n")

print("Defect sites: \n----------------------")
print( "Si in both ring of size 3 and 4 ", perc_def)
print("\n\n")

print(np.where(post_rings[:, 2] >10))

for i in range(len(size)):

    plt.figure()
    plt.title("Rings of size " + str(size[i]))
    plt.hist([ndef_rings[:, i], def_rings[:, i]], alpha = 0.5, color = ["green", "purple"], edgecolor = "k", label = ["Normal sites", "Defect sites"], density = True)
    plt.legend()
    plt.show()
    
    print("x-axis: (defects)", np.unique(def_rings[:,i]))

#%%
print(np.count_nonzero(ndef_rings[:, 0]))
print(np.count_nonzero(def_rings[:, 0]))

print(np.count_nonzero(ndef_rings[:, 0] == 0))
#%%
dihedral_angles, dihedral_idx, defect_idx = ao.analyze_diheds("C:\\Users\\kajah\\git_repo\\Bcs\\xyz_files")
#%%
print(np.array(dihedral_angles).shape, np.array(dihedral_idx).shape, np.array(defect_idx).shape)

#%%
defect_idx = np.array(defect_idx)

post_dihed = dihedral_angles[defect_idx]
def_dihed = dihedral_angles[defect_idx + 72]
ndef_dihed = np.delete(dihedral_angles, np.append(defect_idx, defect_idx+72, axis = 0), axis = 0)

print(post_dihed.shape, def_dihed.shape, ndef_dihed.shape)

#%%

plt.figure()
plt.title("Dihedral angles around normal sites")
plt.hist(ndef_dihed.flatten(), alpha = 0.5, color = "green", edgecolor = "k")
plt.xlabel("Degrees")
plt.ylabel("Occurrence")
plt.show()

plt.figure()
plt.title("Dihedral angles around defects")
plt.hist(def_dihed.flatten(), alpha = 0.5, color = "green", edgecolor = "k")
plt.hist(post_dihed.flatten(), alpha = 0.5, color = "purple", edgecolor = "k")
plt.xlabel("Degrees")
plt.ylabel("Occurrence")
plt.show()

plt.figure()
plt.title("Dihedral angles around normal sites (absolute value)")
plt.hist(np.abs(ndef_dihed.flatten()), alpha = 0.5, color = "green", edgecolor = "k")
plt.xlabel("Degrees")
plt.ylabel("Occurrence")
plt.show()

plt.figure()
plt.title("Dihedral angles around defects (absolute value)")
plt.hist(np.abs(def_dihed.flatten()), alpha = 0.5, color = "green", edgecolor = "k")
plt.hist(np.abs(post_dihed.flatten()), alpha = 0.5, color = "purple", edgecolor = "k")
plt.xlabel("Degrees")
plt.ylabel("Occurrence")
plt.show()

#%%

plt.figure()
plt.title("Dihedral angles around normal sites")
plt.hist(ndef_dihed[:, :12].flatten(), alpha = 0.5, color = "green", edgecolor = "k")
plt.xlabel("Degrees")
plt.ylabel("Occurrence")
plt.show()

plt.figure()
plt.title("Dihedral angles around defects  (first atom in dihedral)")
plt.hist(def_dihed[:, :12].flatten(), alpha = 0.5, color = "green", edgecolor = "k")
plt.hist(post_dihed[:, :12].flatten(), alpha = 0.5, color = "purple", edgecolor = "k")
plt.xlabel("Degrees")
plt.ylabel("Occurrence")
plt.show()


plt.figure()
plt.title("Dihedral angles around defects (second atom in dihedral)")
plt.hist(def_dihed[:, 12:].flatten(), alpha = 0.5, color = "green", edgecolor = "k")
plt.hist(post_dihed[:, 12:].flatten(), alpha = 0.5, color = "purple", edgecolor = "k")
plt.xlabel("Degrees")
plt.ylabel("Occurrence")
plt.show()

plt.figure()
plt.title("Dihedral angles around defects (second atom in dihedral)")
plt.hist(np.abs(def_dihed[:, 12:].flatten()), alpha = 0.5, color = "green", edgecolor = "k")
plt.hist(np.abs(post_dihed[:, 12:].flatten()), alpha = 0.5, color = "purple", edgecolor = "k")
plt.xlabel("Degrees")
plt.ylabel("Occurrence")
plt.show()


plt.figure()
#plt.scatter(def_dihed[:,0], def_dihed[:, 12])
for i in range(58):
    plt.scatter(def_dihed[i, :12], def_dihed[i, 12:])
plt.show()

#%%
def_first_sort = np.sort(def_dihed[:,:12], axis = 1)
def_sec_sort = np.sort(def_dihed[:, 12:], axis = 1)

plt.figure()
plt.plot([-110, -75, 150], [-190, -150, 75])
for i in range(58):
    plt.scatter(def_first_sort[i], def_sec_sort[i], c = "b", alpha = 0.5, s = 70)
plt.show()

#%%
ndef_first_sort = np.sort(ndef_dihed[:,:12], axis = 1)
ndef_sec_sort = np.sort(ndef_dihed[:, 12:], axis = 1)

from collections import Counter

plt.figure()
plt.plot([-110, -75, 150], [-190, -150, 75])
for i in range(10):
    x = list(ndef_first_sort[:,i])
    y =  list(ndef_sec_sort[:, i])
    # count the occurrences of each point
    c = Counter(zip(x, y))
    # create a list of the sizes, here multiplied by 10 for scale
    s = [0.1*c[(xx,yy)] for xx,yy in zip(x, y)]
    

    plt.scatter(x, y, c = "blue", s = 0.1, alpha = 0.2)
plt.show()

#%%
import seaborn as sns

plt.title("Dihedrals, defect sites")
sns.kdeplot(x = def_first_sort.flatten(), y = def_sec_sort.flatten(), cmap="Reds", shade=True)
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

"""
plt.scatter(ndef_first_sort[:,1], ndef_sec_sort[:, 1], c = "b", s = s)
plt.scatter(ndef_first_sort[:,2], ndef_sec_sort[:, 2], c = "b", s = s)
plt.scatter(ndef_first_sort[:,3], ndef_sec_sort[:, 3], c = "b", s = s)
plt.scatter(ndef_first_sort[:,4], ndef_sec_sort[:, 4], c = "b", s = s)
plt.scatter(ndef_first_sort[:,5], ndef_sec_sort[:, 5], c = "b", s = s)
plt.scatter(ndef_first_sort[:,6], ndef_sec_sort[:, 6], c = "b", s = s)
plt.scatter(ndef_first_sort[:,7], ndef_sec_sort[:, 7], c = "b", s = s)
plt.scatter(ndef_first_sort[:,8], ndef_sec_sort[:, 8], c = "b", s = s)
plt.scatter(ndef_first_sort[:,9], ndef_sec_sort[:, 9], c = "b", s = s)
plt.scatter(ndef_first_sort[:,10], ndef_sec_sort[:, 10], c = "b", s = s)
plt.scatter(ndef_first_sort[:,11], ndef_sec_sort[:, 11], c = "b", s = s)
plt.show()
"""

#%%

#%%
"""
plt.scatter(def_dihed[:,1], def_dihed[:, 13], c = np.arange(0,58, 1))
plt.scatter(def_dihed[:,2], def_dihed[:, 14], c = np.arange(0,58, 1))
plt.scatter(def_dihed[:,3], def_dihed[:, 15], c = np.arange(0,58, 1))
plt.scatter(def_dihed[:,4], def_dihed[:, 16], c = np.arange(0,58, 1))
plt.scatter(def_dihed[:,5], def_dihed[:, 17], c = np.arange(0,58, 1))
plt.scatter(def_dihed[:,6], def_dihed[:, 18], c = np.arange(0,58, 1))
plt.scatter(def_dihed[:,7], def_dihed[:, 19], c = np.arange(0,58, 1))
plt.scatter(def_dihed[:,8], def_dihed[:, 20], c = np.arange(0,58, 1))
plt.scatter(def_dihed[:,9], def_dihed[:, 21], c = np.arange(0,58, 1))
plt.scatter(def_dihed[:,10], def_dihed[:, 22], c = np.arange(0,58, 1))
plt.scatter(def_dihed[:,11], def_dihed[:, 23], c = np.arange(0,58, 1))

plt.show()
"""
#%%
"""
plt.figure()
plt.scatter(ndef_dihed[:,0], ndef_dihed[:, 12], c = np.arange(0,27388, 1))
plt.scatter(ndef_dihed[:,1], ndef_dihed[:, 13], c = np.arange(0,27388, 1))
plt.scatter(ndef_dihed[:,2], ndef_dihed[:, 14], c = np.arange(0,27388, 1))
plt.scatter(ndef_dihed[:,3], ndef_dihed[:, 15], c = np.arange(0,27388, 1))
plt.scatter(ndef_dihed[:,4], ndef_dihed[:, 16], c = np.arange(0,27388, 1))
plt.scatter(ndef_dihed[:,5], ndef_dihed[:, 17], c = np.arange(0,27388, 1))
plt.scatter(ndef_dihed[:,6], ndef_dihed[:, 18], c = np.arange(0,27388, 1))
plt.scatter(ndef_dihed[:,7], ndef_dihed[:, 19], c = np.arange(0,27388, 1))
plt.scatter(ndef_dihed[:,8], ndef_dihed[:, 20], c = np.arange(0,27388, 1))
plt.scatter(ndef_dihed[:,9], ndef_dihed[:, 21], c = np.arange(0,27388, 1))
plt.scatter(ndef_dihed[:,10], ndef_dihed[:, 22], c = np.arange(0,27388, 1))
plt.scatter(ndef_dihed[:,11], ndef_dihed[:, 23], c = np.arange(0,27388, 1))
plt.show()
"""
#%%

plt.figure()
plt.scatter(ndef_dihed[:, :12], ndef_dihed[:, 12:])
plt.scatter(def_dihed[:, :12], def_dihed[:, 12:])
plt.show()

#%%

plt.figure()
plt.scatter(def_dihed[:, :12].flatten(), def_dihed[:, 12:].flatten())
plt.show()
#%%
import seaborn as sns

def_dihedrals = np.array([def_dihed[:, :12].flatten(), def_dihed[:, 12:].flatten()]).T
print(def_dihed.shape)


sns.heatmap(def_dihedrals)

#%%
final, defect_idx = ao.analyze_Si("C:\\Users\\kajah\\git_repo\\Bcs\\xyz_files")

#%%
print(final[278])
print(final[279])
print(final.shape)
#%%
print("Number of defects found: ", len(defect_idx))
defect_idx= np.array(defect_idx)

post_idx = defect_idx
def_idx = defect_idx+72
ndef_final = np.delete(final, np.append(def_idx, post_idx), axis = 0)

print(final[post_idx[-3:]])
print(final[def_idx[-3:]])

post_angles = final[post_idx, 2:12]
def_angles = final[def_idx, 2:12]
ndef_angles = ndef_final[:,2:12]

def_bonds = final[def_idx, 12:]
ndef_bonds = ndef_final[:,12:]
post_bonds = final[post_idx, 12:]

print(len(ndef_angles), len(post_angles), len(def_angles), len(final)-len(ndef_angles))

#%%
#OSIO ANgles
def_osio = def_angles[:, 4:].flatten()
ndef_osio = ndef_angles[:, 4:].flatten()
post_osio = post_angles[:, 4:].flatten()

plt.figure()
plt.scatter(np.arange(0,58,1), def_angles[:, 4], c = "b")
plt.scatter(np.arange(0,58,1), def_angles[:, 5], c = "b")
plt.scatter(np.arange(0,58,1), def_angles[:, 6], c = "b")
plt.scatter(np.arange(0,58,1), def_angles[:, 7], c = "b")
plt.scatter(np.arange(0,58,1), def_angles[:, 8], c = "b")
plt.scatter(np.arange(0,58,1), def_angles[:, 9], c = "b")
plt.show()

plt.figure()
plt.scatter(np.arange(0,27388,1), ndef_angles[:, 4], c = "b")
plt.scatter(np.arange(0,27388,1), ndef_angles[:, 5], c = "b")
plt.scatter(np.arange(0,27388,1), ndef_angles[:, 6], c = "b")
plt.scatter(np.arange(0,27388,1), ndef_angles[:, 7], c = "b")
plt.scatter(np.arange(0,27388,1), ndef_angles[:, 8], c = "b")
plt.scatter(np.arange(0,27388,1), ndef_angles[:, 9], c = "b")
plt.show()

plt.figure()
plt.hist(np.sum(def_angles[:, 4:], axis = 1))
plt.show()

plt.figure()
plt.hist(np.sum(ndef_angles[:, 4:], axis = 1))
plt.show()


print(def_angles[:, :6].shape)
plt.figure()
plt.title("O-Si-O angles around defects")
plt.hist(def_osio, alpha = 0.5, color = "green", edgecolor = "k", label = "pre")
plt.hist(post_osio, alpha = 0.5, color = "purple", edgecolor = "k", label = "post")
plt.legend()
plt.show()

print(post_angles[:10,4:])
plt.figure()
plt.title("O-Si-O angles around normal sites")
plt.hist(ndef_osio, alpha = 0.5, color = "green", edgecolor = "k")
plt.show()
#%%
cutoff_angle = 125
cut_offs = np.arange(105, 150, 1)

pre_def = np.count_nonzero(np.count_nonzero(def_angles[:, 4:]>cutoff_angle, axis = 1)>0)
no_def = np.count_nonzero(np.count_nonzero(ndef_angles[:, 4:]>cutoff_angle, axis = 1)>0)
post_def = np.count_nonzero(np.count_nonzero(post_angles[:, 4:]>cutoff_angle, axis = 1)==2)


print(np.array(np.where(post_angles[:, 4:]>cutoff_angle)).T)
print(np.count_nonzero(def_angles[:, 4:]>cutoff_angle, axis = 1))
plt.figure()
plt.title("Si atoms with at least one angle larger than the cut-off angle", size = 16)
plt.plot(cutoff_angle, no_def/len(ndef_angles[:, 4:])*100, color = "royalblue", marker = "o", markersize = 8, label = "normal")
plt.plot(cutoff_angle, pre_def/len(def_angles[:, 4:])*100, color = "orange", marker = "x", markersize = 8, label = "pre defect")
#plt.plot(cutoff_angle, post_def/len(post_angles[:, 4:])*100, "bx", label = "post defect")

for cutoff_angle in cut_offs:
    pre_def = np.count_nonzero(np.count_nonzero(def_angles[:, 4:]>cutoff_angle, axis = 1)>0)
    no_def = np.count_nonzero(np.count_nonzero(ndef_angles[:, 4:]>cutoff_angle, axis = 1)>0)
    #post_def = np.count_nonzero(np.count_nonzero(post_angles[:, 4:]>cutoff_angle, axis = 1)==2)
    
    print("\n\nCut-off angle: ", cutoff_angle)
    print("Normal: Nr of occurrences: ", no_def, ". Percentage : ", np.round(no_def/len(ndef_angles[:, 4:])*100, 2))
    print("Pre defect: Nr of occurrences: ", pre_def, ". Percentage : ", np.round(pre_def/len(def_angles[:, 4:])*100, 2))
    #print("Post defect: Nr of occurrences: ", post_def, ". Percentage : ", np.round(post_def/len(post_angles[:, 4:])*100, 2))
    
    plt.plot(cutoff_angle, no_def/len(ndef_angles[:, 4:])*100, color = "royalblue", marker = "o", markersize = 8)
    plt.plot(cutoff_angle, pre_def/len(def_angles[:, 4:])*100, color = "orange", marker = "x", markersize = 8)

    #plt.plot(cutoff_angle, post_def/len(post_angles[:, 4:])*100, "bx")
plt.ylabel("Percentage", size = 16)
plt.xlabel("Cut-off angle", size = 16)
plt.xticks(size = 14)
plt.yticks(size = 14)
plt.legend(fontsize = 12)
plt.show()
#%%
print(def_rings[:, 1])
def_cut = np.count_nonzero(def_angles[:, 4:]>132, axis = 1)
def_tot = def_rings[:, 1] + def_cut

print(np.count_nonzero(def_tot==2))

ndef_cut = np.count_nonzero(ndef_angles[:, 4:]>132, axis = 1)
ndef_tot = ndef_rings[:, 1] + ndef_cut

print(np.count_nonzero(ndef_tot==2))



#%%

colors = ["red", "blue", "orange", "purple", "green"]
i = 0
cut_offs = np.arange(115, 130, 2)

for angle in cut_offs:
    pre_def = np.count_nonzero(def_angles[:, 4:]>angle, axis = 1)
    no_def = np.count_nonzero(ndef_angles[:, 4:]>angle, axis = 1)
    post_def = np.count_nonzero(post_angles[:, 4:]>angle, axis = 1)
    
    plt.figure()
    plt.title("Percentage of O-Si-O angles larger than " + str(angle))
    #plt.plot(0, np.count_nonzero(pre_def==0)/len(pre_def)*100, "bo", label = "pre defect")
    #plt.plot(0, np.count_nonzero(no_def==0)/len(no_def)*100, "ro", label = "normal")
    #plt.plot(0, np.count_nonzero(post_def==0)/len(post_def)*100, "bx", label = "post defect")
    
    plt.plot(1, np.count_nonzero(pre_def == 1)/len(pre_def)*100, "ro", label = "pre defect")
    plt.plot(1, np.count_nonzero(post_def == 1)/len(post_def)*100, "kx", label = "post defect")
    plt.plot(1, np.count_nonzero(no_def == 1)/len(no_def)*100, "bo", label = "normal")
    
    plt.plot(2, np.count_nonzero(pre_def == 2)/len(pre_def)*100, "ro")
    plt.plot(2, np.count_nonzero(post_def == 2)/len(post_def)*100, "kx")
    plt.plot(2, np.count_nonzero(no_def == 2)/len(no_def)*100, "bo")
    
    plt.legend()
    plt.ylabel("Percentage")
    plt.xlabel("Number of occurrences")
    plt.legend()
    plt.grid()
    i +=1
    plt.show()
    
#%%
#SIOSI angles
def_siosi = def_angles[:, :4].flatten()
ndef_siosi = ndef_angles[:, :4].flatten()
post_siosi = post_angles[:, :4].flatten()

plt.figure()
plt.hist(np.sum(def_angles[:, :4], axis = 1))
plt.show()
plt.figure()
plt.hist(np.sum(ndef_angles[:, :4], axis = 1))
plt.show()



plt.figure()
plt.title("Si-O-Si angles around defects")
plt.hist(def_siosi, alpha = 0.5, color = "green", edgecolor = "k", label = "pre")
plt.hist(post_siosi, alpha = 0.5, color = "purple", edgecolor = "k", label = "post")
plt.legend()
plt.show()


plt.figure()
plt.title("Si-O-Si angles around normal sites")
plt.hist(ndef_siosi, alpha = 0.5, color = "green", edgecolor = "k")
plt.show()
#%%
#Bonds
plt.figure()
plt.title("Bonds around defects")
plt.hist(def_bonds.flatten(), bins = 12, alpha = 0.5, color = "green", edgecolor = "k", label = "pre")
plt.hist(post_bonds.flatten(), bins = 12, alpha = 0.5, color = "purple", edgecolor = "k", label = "post")
plt.legend()
plt.show()

plt.figure()
plt.title("Bonds around normal sites")
plt.hist(ndef_bonds.flatten(), alpha = 0.5, color = "green", edgecolor = "k")
plt.show()

plt.figure()
plt.title("Bonds around defects")
plt.hist(def_bonds[:,0]/def_bonds[:,-1], bins = 12, alpha = 0.5, color = "green", edgecolor = "k", label = "pre")
plt.legend()
plt.show()

plt.figure()
plt.title("Bonds around normal sites")
plt.hist(ndef_bonds[:,0]/ndef_bonds[:,-1], alpha = 0.5, color = "green", edgecolor = "k")
plt.show()



#%%
"""
stats_ndef = []
stats_def = []
stats_post  = []

for i in range(len(def_angles[0])):
    stats_ndef.append([max(ndef_angles[:,i]), min(ndef_angles[:,i]), np.mean(ndef_angles[:,i])])
    stats_def.append([max(def_angles[:,i]), min(def_angles[:,i]), np.mean(def_angles[:,i])])
    stats_post.append([max(post_angles[:,i]), min(post_angles[:,i]), np.mean(post_angles[:,i])])
    
for i in range(len(def_bonds[0])):
    stats_ndef.append([max(ndef_bonds[:,i]), min(ndef_bonds[:,i]), np.mean(ndef_bonds[:,i])])
    stats_def.append([max(def_bonds[:,i]), min(def_bonds[:,i]), np.mean(def_bonds[:,i])])
    stats_post.append([max(post_bonds[:,i]), min(post_bonds[:,i]), np.mean(post_bonds[:,i])])

stats_ndef = np.array(stats_ndef)
stats_def = np.array(stats_def)
stats_post = np.array(stats_post)

#%%

all_stats = pd.DataFrame(np.append(stats_ndef, np.append(stats_def, stats_post, axis = 1), axis = 1), columns = ["Max, normal", "Min, normal", "Average, normal", "Max, normal", "Min, defect (pre)", "Average, defect (pre)", "Max, defect (post)", "Min, defect (post)", "Average, defect (post)"])

all_stats.to_excel("C:\\Users\\kajah\\git_repo\\Bcs\\stats_Siatoms.xlsx")
"""
#%%
for i in range(len(def_angles[0])):
    print("The ", i+1, "-th largest angles: \n-----------------------\n\nFor defects:\nMaximum: ", max(def_angles[:,i]), "\nMinimum: ", min(def_angles[:, i]), "\nAverage: ", np.mean(def_angles[:,i]))
    print("\n\nFor normals:\nMaximum: ", max(ndef_angles[:,i]), "\nMinimum: ", min(ndef_angles[:, i]), "\nAverage: ", np.mean(ndef_angles[:,i]), "\n\n\n")
#%%
for i in range(len(def_bonds[0])):
    print("The ", i+1, "-th largest bonds: \n-----------------------\n\nFor defects:\nMaximum: ", max(def_bonds[:,i]), "\nMinimum: ", min(def_bonds[:, i]), "\nAverage: ", np.mean(def_bonds[:,i]))
    print("\n\nFor normals:\nMaximum: ", max(ndef_bonds[:,i]), "\nMinimum: ", min(ndef_bonds[:, i]), "\nAverage: ", np.mean(ndef_bonds[:,i]), "\n\n\n")


#%%
print(final[0])
post = np.array([378, 372, 358, 353, 347, 345, 339, 335, 332, 328, 326, 323, 319, 307, 304, 300, 295, 287, 275, 244,
        242, 239, 233, 225, 219, 217, 213, 209, 204, 202, 197, 193, 183, 172, 162, 160, 158, 155, 148, 143,
        138, 114, 111, 106, 99, 97, 94, 88, 78, 75, 73, 71, 63, 60, 41, 25, 11, 3])
defect = post-1


#%%
avg_angles = np.mean(final[:, 2:10], axis = 1)
max_angles = final[:, 2]

def_idx = np.where(is_defect == 1)
ndef_idx = np.where(is_defect == 0)
def_idx = np.intersect1d(final[:, 1], def_idx, return_indices = True)[1]
ndef_idx = np.intersect1d(final[:, 1], ndef_idx, return_indices = True)[1]

def_final = final[def_idx]
ndef_final = final[ndef_idx]

print(final.shape)
print(def_final.shape)
print(ndef_final.shape)

print(min(max_angles[def_idx]))

plt.figure()
plt.hist(max_angles[ndef_idx], alpha = 0.5, label = "Normal")
#plt.hist(max_angles[def_idx], alpha = 0.5, label = "defects")
#plt.xlim(130, 180)
plt.legend()
plt.show()

print(max(max_angles[def_idx]))

print(len(max_angles[def_idx][max_angles[def_idx]>140]))

plt.figure()
plt.hist(avg_angles[def_idx])

plt.hist(avg_angles[ndef_idx])
plt.show()

#%%

stat, df = ao.generate_data("C:\\Users\\kajah\\git_repo\\Bcs\\xyz_files", "C:\\Users\\kajah\\git_repo\\Bcs")
#%%
post = [378, 372, 358, 353, 347, 345, 339, 335, 332, 328, 326, 323, 319, 307, 304, 300, 295, 287, 275, 244,
        242, 239, 233, 225, 219, 217, 213, 209, 204, 202, 197, 193, 183, 172, 162, 160, 158, 155, 148, 143,
        138, 114, 111, 106, 99, 97, 94, 88, 78, 75, 73, 71, 63, 60, 41, 25, 11, 3]
i = 0

nodef_osio_a = []
def_osio_a = []
nodef_osio_b = []
def_osio_b = []

nodef_siosi_a = []
def_siosi_a = []
nodef_siosi_b = []
def_siosi_b = []

def_boxid = []
nodef_boxid = []

for dic in stat:
    if i + 1 in post:
        def_osio_a.append(dic["osio angles"])
        def_osio_b.append(dic["osio bondlength"])
        def_siosi_a.append(dic["siosi angles"])
        def_siosi_b.append(dic["siosi bondlength"])
        
        def_boxid.append(i)
        
    elif i not in post:
        nodef_osio_a.append(dic["osio angles"])
        nodef_siosi_a.append(dic["siosi angles"])
        nodef_osio_b.append(dic["osio bondlength"])
        nodef_siosi_b.append(dic["siosi bondlength"])
        
        nodef_boxid.append(i)
    i +=1
#%%

def plotter(dx, dy, ndx, ndy, titles):
    for i in range(len(dy[0])):
        plt.figure()
        plt.title(titles[i])
        plt.plot(dx, dy[:,i], "ro", label = "Defect" )
        plt.plot(ndx, ndy[:,i], "bo", label = "No defect")
        plt.legend()
        plt.show()


titles = ["O-Si-O bonds, average angle", "O-Si-O bonds, max angle", "O-Si-O bonds, min angle"]
plotter(np.array(def_boxid), np.array(def_osio_a), np.array(nodef_boxid), np.array(nodef_osio_a), titles)

titles = ["O-Si-O bonds, average bond", "O-Si-O bonds, max bond", "O-Si-O bonds, min bond"]
plotter(np.array(def_boxid), np.array(def_osio_b), np.array(nodef_boxid), np.array(nodef_osio_b), titles)

titles = ["Si-O-Si bonds, average angle", "Si-O-Si bonds, max angle", "Si-O-Si bonds, min angle"]
plotter(np.array(def_boxid), np.array(def_siosi_a), np.array(nodef_boxid), np.array(nodef_siosi_a), titles)

titles = ["Si-O-Si bonds, average bond", "Si-O-Si bonds, max bond", "Si-O-Si bonds, min bond"]
plotter(np.array(def_boxid), np.array(def_siosi_b), np.array(nodef_boxid), np.array(nodef_siosi_b), titles)  
        
print("Statistics:\n-------------------------------- ")
print("Defects: \n Angles:\nAvg: ", np.mean(np.array(def_osio_a)))
        
#%%


        

"""
pre_boxid = []
post_boxid = []
post_avg = []
pre_avg = []
post_max = []
pre_max = []
post_min = []
pre_min = []



post_avgbond = []
pre_avgbond = []
post_maxbond = []
pre_maxbond = []
post_minbond = []
pre_minbond = []


apost_avg = []
apost_max = []
apost_min = []
apost_avgbond = []
apost_maxbond = []
apost_minbond = []

for dic in stat:
    
    if i+1 in post:
        post_avg.append(dic["avg_angle"])
        post_max.append(dic["max_angle"])
        post_min.append(dic["min_angle"])
        post_avgbond.append(dic["avg_bondlength"])
        post_maxbond.append(dic["max_bondlength"])
        post_minbond.append(dic["min_bondlength"])
        
        apost_avg.append(stat[i]["avg_angle"])
        apost_max.append(stat[i]["max_angle"])
        apost_min.append(stat[i]["min_angle"])
        apost_avgbond.append(stat[i]["avg_bondlength"])
        apost_maxbond.append(stat[i]["max_bondlength"])
        apost_minbond.append(stat[i]["min_bondlength"])
        
        post_boxid.append(i)
    elif i not in post:
        pre_avg.append(dic["avg_angle"])
        pre_max.append(dic["max_angle"])
        pre_min.append(dic["min_angle"])
        pre_avgbond.append(dic["avg_bondlength"])
        pre_maxbond.append(dic["max_bondlength"])
        pre_minbond.append(dic["min_bondlength"])
        
        pre_boxid.append(i)
    i +=1
"""
#%%
"""
pid = post[::-1]
plt.figure()
plt.title("Average angle")
plt.plot(post_boxid, post_avg, "ro", label = "pre box with trap")
plt.plot(pre_boxid, pre_avg, "bo", label = "Normal box")
#plt.plot(pid, apost_avg, "go", label = "post box with trap")
#plt.legend()
plt.show()

plt.figure()
plt.title("Maximum angle")
plt.plot(post_boxid, post_max, "ro", label = "Box with defect")
plt.plot(pre_boxid, pre_max, "bo", label = "Normal box")
plt.show()

plt.figure()
plt.title("Minimum angle")
plt.plot(post_boxid, post_min, "ro", label = "Box with defect")
plt.plot(pre_boxid, pre_min, "bo", label = "Normal box")
plt.show()

plt.figure()
plt.title("Angle fraction")
plt.plot(post_boxid, np.array(post_max)/np.array(post_min), "ro")
plt.plot(pre_boxid, np.array(pre_max)/np.array(pre_min), "bo")
plt.show()

print("Angles")
print("Defects:\nAverage ", np.mean(np.array(post_avg)), "\nMaximum: ", np.mean(np.array(post_max)), "\nMinimum: ", np.array(np.mean(post_min)))
print("\nNormal:\nAverage ", np.mean(np.array(pre_avg)), "\nMaximum: ", np.mean(np.array(pre_max)), "\nMinimum: ", np.array(np.mean(pre_min)))
#%%
plt.figure()
plt.title("Average bondlength")
plt.plot(post_boxid, post_avgbond, "ro", label = "Box with defect")
plt.plot(pre_boxid, pre_avgbond, "bo", label = "Normal box")
plt.show()

plt.figure()
plt.title("Maximum bondlength")
plt.plot(post_boxid, post_maxbond, "ro", label = "Box with defect")
plt.plot(pre_boxid, pre_maxbond, "bo", label = "Normal box")
plt.show()

plt.figure()
plt.title("Minimum bondlength")
plt.plot(post_boxid, post_minbond, "ro", label = "Box with defect")
plt.plot(pre_boxid, pre_minbond, "bo", label = "Normal box")
plt.show()

print("\n\nBondlengths")
print("Defects:\nAverage ", np.mean(np.array(post_avgbond)), "\nMaximum: ", np.mean(np.array(post_maxbond)), "\nMinimum: ", np.array(np.mean(post_minbond)))
print("\nNormal:\nAverage ", np.mean(np.array(pre_avgbond)), "\nMaximum: ", np.mean(np.array(pre_maxbond)), "\nMinimum: ", np.array(np.mean(pre_minbond)))
"""

#%%
print(df.head())
i = 0
for dic in stat:
    print(dic["name"], i)
    i += 1

#%%
"""
plt.figure()
plt.title("Dihedral angles", size = 20)
plt.hist(dihedral_angles)
plt.xlabel("Angle (Degrees)", size = 18)
plt.xticks(size = 18)
plt.yticks(size = 18)
plt.show()



plt.figure()
plt.title("O-Si-O angle distribution", size = 20)
plt.hist(osio_angles, bins = 10)
plt.xlabel("Angle (Degrees)", size = 18)
plt.xticks(size = 18)
plt.yticks(size = 18)
plt.show()

plt.figure()
plt.title("Si-O-Si angle distribution", size = 20)
plt.hist(siosi_angles)
plt.xlabel("Angle (Degrees)", size = 18)
plt.xticks(size = 18)
plt.yticks(size = 18)
plt.show()

print(min(siosi_angles), siosi_bonds2[np.where(siosi_angles == min(siosi_angles))])
print(np.where(siosi_angles == min(siosi_angles)))
print(siosi_angles[:5], siosi_bonds2[:5])
#%%
idx = np.array(np.where(siosi_angles == min(siosi_angles)))[0]
for dic in stat:
    print(dic["name"])
    print("------------------------------")
    print("Nr. of bonds: ", dic["bonds"])
    print("Nr. of angles: ", dic["angles"])
    print("Nr. of dihedrals: ", dic["dihedral angles"])
    idx -= dic["angles"]
    if idx[0]>0:
        print(idx)
    else:
        break
    print("------------------------------")
    print("------------------------------")
"""
#%%
cutoffs = [3.39, 2.0, 3.0] 
#cut off lengths for bonds, unit Å. Distances between 
#atoms that are shorter or equal to these are chemical bonds
#or next-nearest neighbor (one atom and two chemical bonds in between).
#First element is for Si Si bonds, second for Si O bonds, last for O O bonds. 

index, bond_lengths, dr = ao.make_bonds(etrap.xArray, etrap.yArray, etrap.zArray, etrap.nameArray, 
                                        etrap.boxX, etrap.boxY, etrap.boxZ, cutoffs)

##########################################################################
#Bond angles and dihedral angles
sisisi_idx, siosi_idx, osio_idx = ao.match_bonds(index)

print(np.array(np.where(osio_idx)).T)

#print(np.array(np.where(siosi_idx)).T)
#########################


"""

#Finding dihedral angles:
start = time.time()
print("\n\nFinding dihedral angles....")
dihedrals = ao.get_dihedrals_fast(a2, index)
print(len(dihedrals))
end = time.time()
print("Time used to find dihedrals: ", end-start)
dihed_angles = ao.calc_dihedral(np.array(dihedrals), dr)
print(dihed_angles.shape)
print(np.array([dihedrals[:10], dihed_angles[:10]]).T)



#########################
#Finding Si-Si rings up to size 6:


print("\n\nFinding rings of size 3")
start = time.time()
sisi_idx = (index == 3)
rings3 = ao.three_ring(a1, sisi_idx) 
print(len(rings3))
end = time.time()
print("Time used: ", end-start)

print("\n\nFinding rings of size 4")
start = time.time()
rings4 = ao.four_ring(rings3, a1, sisi_idx)
print(len(rings4))
end = time.time()
print("Time used: ", end-start)

print("\n\nFinding rings of size 5")
start = time.time()
rings5 = ao.five_ring(rings3, rings4, a1, sisi_idx)
print(len(rings5))
end = time.time()
print("Time used: ", end-start)

print("\n\nFinding rings of size 6")
start = time.time()
rings6 = ao.six_ring(rings3, rings4, rings5, a1) 
print(len(rings6))
end = time.time()
print("Time used: ", end-start)
"""
# %%
