# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 11:28:13 2022

@author: kajah
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import PDF as pdf
import AmorphousOxide as ao
import time

xyz_path = "C:\\Users\\kajah\\ntnu\\bachelor_oppg\\xyz_files\\etrap_015.xyz"
csv_path = "C:\\Users\\kajah\\git_repo\\Bcs\\python code\\data\\"

etrap015 = ao.AmorphousOxide(216, 72, 144, [14.949, 14.949, 14.949], xyz_path)
cutoffs = [3.5, 2.0, 3.0] #sisi, sio, oo

#print(len(etrap015.nameArray))


start = time.time()
#df = etrap015.get_chain2(cutoffs)

df = etrap015.get_dihedral_angles(cutoffs)
df.to_csv(csv_path + "dihed_new.csv")
#%%
print(df)

#%%

#df2 = etrap015.get_chain3_2(df, df)
end = time.time()
print("Time passed: ", end - start)
#print(df2)
#df2.to_csv(csv_path + "angle_bonds.csv")

#%%
"""

csv_path = "C:\\Users\\kajah\\git_repo\\Bcs\\python code\\data\\"

df = pd.read_table(xyz_path, skiprows=2, delim_whitespace=True, nrows = 216, names=['atom', 'x', 'y', 'z'])
#xyz_df = df[["x", "y", "z"]]
#print(xyz_df.loc[0].values)

#bz = 0.008
#etrap = pdf.PDF(bz)

#angles = pd.read_csv(csv_path + "angles.csv")

#df = pd.read_csv(csv_path + "single_bonds.csv")
#sio_bonds = df.loc[df["name"] == "Si O"]
#sio_bonds = sio_bonds.append(df.loc[df["name"] == "O Si"])
#sio_bonds = sio_bonds[["dr", "index", "name"]]


#df_dihedral = pdf.get_dihedrals(sio_bonds, angles[["Index", "Bond length 1", "Bond length 2", "Length 3", "Bond name"]], xyz_path, etrap)
#df_dihedral = pdf.remove_rev_dupes(df_dihedral)
#df_dihedral.to_csv(csv_path + "dihedral_angles.csv")

pdf.get_ML_data(xyz_path, csv_path)


angles = pd.read_csv(csv_path + "angles.csv")
dihedral = pd.read_csv(csv_path + "dihedral_angles.csv")

"""
"""
RINGS:
all bond lengths must be smaller than the cutoff 3.2/3.5
All rings that contain all the same atoms as another ring must not be counted!

"""
#%%
"""
plt.figure()
plt.title("Dihedral angles")
plt.hist(dihedral["Dihedral angle"])
plt.xlabel("Degrees")
plt.show()
"""
#%%
"""

etrap = pdf.PDF()
etrap.calculate_pdf(xyz_path, csv_path)

#%%
test = pdf.PDF(bz = 0.01, max_r = 2)
test.get_bonds("C:\\Users\\kajah\\ntnu\\bachelor_oppg\\xyz_files\\etrap_015.xyz", "C:\\Users\\kajah\\git_repo\\Bcs\\python code\\data\\test.csv")


# %%


df = pd.read_csv(csv_path)


plot_sisi = df[["dr, Si-Si", "Si-Si"]].sort_values(by=["dr, Si-Si"])
#plot_sisi = plot_sisi.loc[(plot_sisi != 0).any(axis=1)]
plt.figure()
plt.title("PDF, Si-Si bonds, boxsize = 14.949")
plt.plot(plot_sisi["dr, Si-Si"], plot_sisi["Si-Si"])
plt.xlabel("distance dr, Å")
# plt.xlim(2.65,3.38)
# plt.ylim(0,2)
plt.show()

plot_sio = df[["dr, Si-O", "Si-O"]].sort_values(by=["dr, Si-O"])
#plot_sio = plot_sio.loc[(plot_sio != 0).any(axis=1)]
plt.figure()
plt.title("PDF, Si-O bonds, boxsize = 14.949")
plt.plot(plot_sio["dr, Si-O"], plot_sio["Si-O"])
#plt.plot(df["distancesio"], df["SiO"].values, "-")
#plt.xlim(1.55, 1.7)
plt.xlabel("distance dr, Å")
plt.show()

plot_oo = df[["dr, O-O", "O-O"]].sort_values(by=["dr, O-O"])
#plot_oo = plot_oo.loc[(plot_oo != 0).any(axis=1)]
plt.figure()
plt.title("PDF, O-O bonds, boxsize = 14.949")
plt.plot(plot_oo["dr, O-O"], plot_oo["O-O"])
#plt.plot(df["distancesio"], df["SiO"].values, "-")
#plt.xlim(1.55, 1.7)
plt.xlabel("distance dr, Å")
plt.show()


maxSio = max(df["Si-O"])
idx = df.index[df['Si-O'] == maxSio]
print("Peak for Si-O bonds: ", maxSio,
      "\nWith distance: ", df["dr, Si-O"].loc[idx])

#print(df[["dr, Si-Si"]].value_counts())
#print(df[["dr, Si-O"]].value_counts())
#print(df[["dr, O-O"]].value_counts())
print(len(plot_sisi) + len(plot_sio) + len(plot_oo))

# %%
#import Analyze

#a = Analyze.Analyze(df)
#bond_labels = a.get_value(4.985916)
#tol = 1e-6

# print("\n\n")
#print("Bond: ", bond_labels["Bond"].item())
#print("Label: ", bond_labels["Label"].item())

# %%
plot_sio = df[["dr, Si-O", "Si-O"]].sort_values(by=["dr, Si-O"])
plot_sio = plot_sio.loc[(plot_sio != 0).any(axis=1)]
plt.figure()
plt.title("PDF, Si-O bonds, boxsize = 14.949")
plt.plot(plot_sio["dr, Si-O"], plot_sio["Si-O"])
#plt.plot(df["distancesio"], df["SiO"].values, "-")
plt.xlim(1.55, 1.85)
plt.xlabel("distance dr, Å")
plt.show()

#print(df[["dr, Si-O"]].loc[(1.55 <= df[["dr, Si-O"]]) & (df[["dr, Si-O"]]<= 1.85)])

#print(plot_sio.loc[plot_sio["dr, Si-O"] >= 1.55])
#print(plot_sio.loc[plot_sio["dr, Si-O"] <= 1.85])

lbldf = df[["dr, Si-O", "Si-O", "Si-O label"]]
print(lbldf.loc[(lbldf["dr, Si-O"] >= 1.55) & (lbldf["dr, Si-O"] <= 1.85)])


# %%
xyz_path = "C:\\Users\\kajah\\ntnu\\bachelor_oppg\\xyz_files\\etrap_015.xyz"
df = pdf.construct_df(xyz_path, max_r = 2)
print("SE UNDER HER:")
#%%
df.to_csv("double_bond.csv")
# %%

df = pd.read_csv("double_bond.csv")
#%%
osio = df.loc[df["Bond name"] == "O-Si-O"]#.reset_index(drop = True)
print(len(osio))
#osio = osio.loc[osio["Bond length 1"] <= 3.5]#.reset_index(drop = True)
#osio = osio.loc[osio["Bond length 2"] <= 3.5]#.reset_index(drop = True)

#osio_angles = osio["Bond Angle"].loc[osio["Bond length 1"] <= 2.5]



siosi = df.loc[df["Bond name"] == "Si-O-Si"]
#siosi = siosi.loc[siosi["Bond length 1"] <= 3.5]#.reset_index(drop = True)
#siosi = siosi.loc[siosi["Bond length 2"] <= 3.5]#.reset_index(drop = True)


plt.figure()
plt.title("O-Si-O bond angles")
plt.hist(np.rad2deg(osio["Bond Angle"]))
plt.xlabel("Degrees")
plt.show()

plt.figure()
plt.title("Si-O-Si bond angles")
plt.hist(np.rad2deg(siosi["Bond Angle"]))
plt.xlabel("Degrees")
plt.show()

print(len(osio) + len(siosi))
df = pd.read_csv("double_bond.csv")
#rev_idx = list(df.loc[:,"Index"].apply(lambda x: " ".join(x[1:-1:].split(", ")[::-1])))
#idx = list(df.loc[:,"Index"].apply(lambda x: " ".join(x[1:-1].split(", "))))
rev_idx = df.loc[:, "Index"].apply(lambda x: " ".join(x.split(" ")[::-1]))
idx = df.loc[:, "Index"]

atoms = np.zeros(216)

for bond in df["Index"].loc[df["Bond name"] == "O-Si-O"]:
    
    lst = bond.split(" ")
    n1, n2, n3 = int(lst[0]), int(lst[1]), int(lst[2])
    #atoms[[n1, n3]] += 1
    atoms[n2] +=1
print(pd.Series(atoms).value_counts())
#for i in range(len(atoms)):
#    print(i, ": ", atoms[i])

#print(idx)
#print(rev_idx)
#print(((rev_idx == idx) == True).any())

#print("(10, 2)"[1:-1].split(", ") == "(2, 10)"[1:-1].split().reverse())

# %%
"""