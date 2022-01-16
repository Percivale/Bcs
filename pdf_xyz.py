# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 14:29:25 2022

@author: kajah
"""

"""
Please take a look at the attached program that was written with archaic Fortran77 by me. 
It calculates partial distribution functions by taking into account (cubic) periodic boundary conditions. 
As a start, make you own code (python?) for the same purpose and calculate PDFs for the structures
that David sent. You will need to modify the number of atoms, their labels, number of frames (= 1),
and the box size accordingly.


The default bin size is probably too narrow (lots of noise) and you can make it thicker. 
Note that PDFs are average properties and will look very similar for the different snapshots.
"""

import pandas as pd
import numpy as np

nmax = 10000
nhdim = 930

au = 0.529177       # Bohr radius
boxx = 35.17239*au  # Box size
boxy = 35.17239*au 
boxz = 35.17239*au
beta = 90           # Box angle (cubic)
pi = np.arccos(-1)   # np.pi ?
bz = 0.01           # Bin size for PDFs (this was too narrow?)
niter = 1           # Number of frames in the xyz file
rcc = 9.3           #max. distance = 0.5*box

def read_xyz(inputfile):
    molecule = pd.read_table(inputfile, skiprows=2, delim_whitespace=True, nrows = 216, names=['atom', 'x', 'y', 'z'])
    return molecule

a_sio2_10 = read_xyz("C:\\Users\\kajah\\ntnu\\bachelor_oppg\\xyz_files\\a_sio2-pos-Replica_nr_01-1.xyz")



atoms = a_sio2_10["atom"].value_counts()

nSi = atoms["Si"]     # nr of Si
nO = atoms["O"]       # nr of O 
nions = nSi + nO      # nr of atoms

#print(nSi, nO, nions)


adhoc1 = boxx*boxy*boxz/(nSi*(nSi-1))/bz            #Cofactors for each PDF
adhoc3 = boxx*boxy*boxz/(nSi*nO - (nSi + nO))/bz
adhoc7 = boxx*boxy*boxz/(nO*nSi - (nO + nSi))/bz
adhoc9 = boxx*boxy*boxz/(nO*(nO-1))/bz
adhocc = boxx*boxy*boxz/(nions*(nions-1))/bz

sp = a_sio2_10["atom"].values
x = a_sio2_10["x"].values
y = a_sio2_10["y"].values
z = a_sio2_10["z"].values

x = x-boxx*np.rint(x/boxx)
y = y-boxy*np.rint(y/boxy)
z = z-boxz*np.rint(z/boxz)

sisi = np.zeros(nhdim)      #Si-Si bonds?
oo = np.zeros(nhdim)        #O-O bonds?
sio = np.zeros(nhdim)       #Si-O bonds?
osi = np.zeros(nhdim)       #O-Si bonds? Why does the sequence matter?
aa = np.zeros(nhdim)        # no idea

for i in range(nions):
    for j in range(nions):
        dx = x[i] - x[j] - boxx*np.rint((x[i]-x[j])/boxx)
        dy = y[i] - y[j] - boxy*np.rint((y[i]-y[j])/boxy)
        dz = z[i] - z[j] - boxz*np.rint((z[i]-z[j])/boxz)
        
        dr = np.sqrt(dx**2 + dy**2 + dz**2)
        
        #Si
        if dr <= rcc:
            ll = np.rint(dr/bz)
            if sp[i] == "Si" and sp[j] == "Si" and i != j:
                sisi[ll] = sisi[ll] + 1
                
            elif sp[i] == "Si" and sp[j] == "O":
                sio[ll] = sio[ll] + 1
            
            elif sp[i] == "O" and sp[J] == "Si":









