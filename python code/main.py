# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 11:28:13 2022

@author: kajah
"""

#%%
import PDF as pdf
xyz_path = "C:\\Users\\kajah\\ntnu\\bachelor_oppg\\xyz_files\\etrap_structures.tar"
csv_path = "data\\etrap_struct.csv"

etrap = pdf.PDF()
etrap.calculate_pdf(xyz_path, csv_path)

#%%

import pandas as pd
import matplotlib.pyplot as plt

csv_path = "etrap_struct.csv"

df = pd.read_csv(csv_path)

#df = df.dropna()
#df = df[["dr, Si-Si", "dr, Si-O","dr, O-O","Si-Si", "Si-O", "O-O"]]
#df = df.loc[(df != 0).any(axis=1)]

#%%
plot_sisi = df[["dr, Si-Si", "Si-Si"]].sort_values(by = ["dr, Si-Si"])
#plot_sisi = plot_sisi.loc[(plot_sisi != 0).any(axis=1)]
plt.figure()
plt.title("PDF, Si-Si bonds, boxsize = 14.949")
plt.plot(plot_sisi["dr, Si-Si"], plot_sisi["Si-Si"])
plt.xlabel("distance dr, Å")
#plt.xlim(2.65,3.38)
#plt.ylim(0,2)
plt.show()

plot_sio = df[["dr, Si-O", "Si-O"]].sort_values(by = ["dr, Si-O"])
#plot_sio = plot_sio.loc[(plot_sio != 0).any(axis=1)]
plt.figure()
plt.title("PDF, Si-O bonds, boxsize = 14.949")
plt.plot(plot_sio["dr, Si-O"], plot_sio["Si-O"])
#plt.plot(df["distancesio"], df["SiO"].values, "-")
#plt.xlim(1.55, 1.7)
plt.xlabel("distance dr, Å")
plt.show()

plot_oo = df[["dr, O-O", "O-O"]].sort_values(by = ["dr, O-O"])
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
print("Peak for Si-O bonds: ", maxSio, "\nWith distance: ", df["dr, Si-O"].loc[idx])

#print(df[["dr, Si-Si"]].value_counts())
#print(df[["dr, Si-O"]].value_counts())
#print(df[["dr, O-O"]].value_counts())

#%%
import Analyze

a = Analyze.Analyze(df)
bond_labels = a.get_value(4.985916)
tol = 1e-6

print("\n\n")
print("Bond: ", bond_labels["Bond"].item())
print("Label: ", bond_labels["Label"].item())

#%%
plot_sio = df[["dr, Si-O", "Si-O"]].sort_values(by = ["dr, Si-O"])
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


#%%


