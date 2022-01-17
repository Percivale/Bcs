# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 11:28:13 2022

@author: kajah
"""
#%%
import pdf_xyz as pdf

xyz_path = "C:\\Users\\kajah\\ntnu\\bachelor_oppg\\xyz_files\\a_sio2-pos-Replica_nr_01-1.xyz"

csv_path = "pdf.csv"

file1 = pdf.PDF()

file1.calculate_pdf(xyz_path, csv_path)

xyz2_path = "C:\\Users\\kajah\\ntnu\\bachelor_oppg\\xyz_files\\a_sio2-pos-Replica_nr_10-1.xyz"
csv2_path = "pdf10.csv"

file2 = pdf.PDF()
file2.calculate_pdf(xyz2_path, csv2_path)

#%%

import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(csv_path)

df = df.dropna()
df = df[["distance", "SiSi", "SiO"]]
df = df.loc[(df != 0).any(axis=1)]

df2 = pd.read_csv(csv2_path)

df2 = df2.dropna()
df2 = df2[["distance", "SiSi", "SiO"]]
df2 = df2.loc[(df2 != 0).any(axis=1)]


#%%


plt.figure()
plt.title("PDF, Si-Si bonds")
plt.plot(df["distance"], df["SiSi"].values, "o", label = "a_sio2-pos-Replica_nr_01-1.xyz")
plt.plot(df2["distance"], df2["SiSi"].values, "o", label = "a_sio2-pos-Replica_nr_10-1.xyz")

#plt.plot(df.index, df["SiSi"].values, "o-", label = "a_sio2-pos-Replica_nr_01-1.xyz")
#plt.plot(df2.index, df2["SiSi"].values, "o-", label = "a_sio2-pos-Replica_nr_10-1.xyz")

plt.legend()
plt.show()


plt.figure()
plt.title("PDF, Si-O bonds")
#plt.plot(df.index, df["SiO"].values,"o", label = "a_sio2-pos-Replica_nr_01-1.xyz")
#plt.plot(df2.index, df2["SiO"].values,"o", label = "a_sio2-pos-Replica_nr_10-1.xyz")

plt.plot(df["distance"], df["SiO"].values, "o", label = "a_sio2-pos-Replica_nr_01-1.xyz")
plt.plot(df2["distance"], df2["SiO"].values, "o", label = "a_sio2-pos-Replica_nr_10-1.xyz")
plt.legend()
plt.show()



"""
plt.figure()
plt.title("rbz")
plt.plot(df["rbz"], df["SiO"].values, "o", label = "a_sio2-pos-Replica_nr_01-1.xyz")
plt.plot(df2["rbz"], df2["SiO"].values, "o", label = "a_sio2-pos-Replica_nr_10-1.xyz")
plt.legend()
plt.show()
"""
#%%

