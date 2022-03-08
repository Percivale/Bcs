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


xyz_path = "C:\\Users\\kajah\\ntnu\\bachelor_oppg\\xyz_files\\etrap_015.xyz"

etrap = ao.AmorphousOxide(216, 72, 144, xyz_path)

stat, df = ao.generate_data("C:\\Users\\kajah\\git_repo\\Bcs\\xyz_files", "C:\\Users\\kajah\\git_repo\\Bcs")

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
