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



class PDF:
    
    def __init__(self):
        self.nmax = 10000
        self.nhdim = 930
        
        self.au = 0.529177              # Bohr radius
        self.boxx = 35.17239*self.au    # Box size
        self.boxy = 35.17239*self.au 
        self.boxz = 35.17239*self.au
        self.beta = 90                  # Box angle (cubic)
        self.pi = np.arccos(-1)         # np.pi ?
        #bz = 0.01                      # Bin size for PDFs (this was too narrow?)
        self.bz = 0.12
        #self.niter = 1                  # Number of frames in the xyz file
        #self.rcc = 9.3                  #max. distance = 0.5*box
    
    
    def read_xyz(self, inputfile):
        molecule = pd.read_table(inputfile, skiprows=2, delim_whitespace=True, nrows = 216, names=['atom', 'x', 'y', 'z'])
        return molecule
    
    
    
    def calculate_pdf(self, xyz_path, csv_path):
        a_sio2 = self.read_xyz(xyz_path)
        
        atoms = a_sio2["atom"].value_counts()
        
        nSi = atoms["Si"]     # nr of Si
        nO = atoms["O"]       # nr of O 
        nions = nSi + nO      # nr of atoms
        

        sp = a_sio2["atom"].values
        x = a_sio2["x"].values
        y = a_sio2["y"].values
        z = a_sio2["z"].values
        
        self.boxx = abs(max(x) - min(x))
        self.boxy = abs(max(y) - min(y))
        self.boxz = abs(max(z) - min(z))
        self.rcc = self.boxx/2
        
        print("Dimensions: ", self.boxx, self.boxy, self.boxz)
        print("Maximumm bond length: ", self.rcc)
        
        
        adhoc1 = self.boxx*self.boxy*self.boxz/(nSi*(nSi-1))/self.bz            #Cofactors for each PDF
        adhoc3 = self.boxx*self.boxy*self.boxz/(nSi*nO - (nSi + nO))/self.bz
        #adhoc7 = self.boxx*self.boxy*self.boxz/(nO*nSi - (nO + nSi))/self.bz
        #adhoc9 = self.boxx*self.boxy*self.boxz/(nO*(nO-1))/self.bz
        #adhocc = self.boxx*self.boxy*self.boxz/(nions*(nions-1))/self.bz

        x = x-self.boxx*np.rint(x/self.boxx)
        y = y-self.boxy*np.rint(y/self.boxy)
        z = z-self.boxz*np.rint(z/self.boxz)

        sisi = np.zeros(self.nhdim)      #Si-Si bonds?
        #oo = np.zeros(self.nhdim)        #O-O bonds?
        sio = np.zeros(self.nhdim)       #Si-O bonds?
        #osi = np.zeros(self.nhdim)       #O-Si bonds? Why does the sequence matter?
        #aa = np.zeros(self.nhdim)        # no idea
        
        distance = np.zeros(self.nhdim)

        for i in range(nions):
            for j in range(nions):
                dx = x[i] - x[j] - self.boxx*np.rint((x[i]-x[j])/self.boxx)
                dy = y[i] - y[j] - self.boxy*np.rint((y[i]-y[j])/self.boxy)
                dz = z[i] - z[j] - self.boxz*np.rint((z[i]-z[j])/self.boxz)
                
                dr = np.sqrt(dx**2 + dy**2 + dz**2)
                
                
                #Si
                if dr <= self.rcc:
                    ll = int(np.rint(dr/self.bz))
                    
                    if sp[i] == "Si" and sp[j] == "Si" and i != j:
                        sisi[ll] = sisi[ll] + 1
                        distance[ll] = dr
                        
                    elif sp[i] == "Si" and sp[j] == "O":
                        sio[ll] = sio[ll] + 1
                        distance[ll] = dr
                    """
                    elif sp[i] == "O" and sp[j] == "Si":
                        osi[ll] == osi[ll] + 1
                        distance[ll] = dr
                    
                    elif sp[i] == "O" and sp[j] == "O":
                        oo[ll] = oo[ll] + 1
                        distance[ll] = dr
                    
                    elif i !=j:
                        aa[ll] = aa[ll] + 1
                        distance[ll] = dr
                    """
        #---------------------------------------------------------------------
        ss = np.zeros(self.nhdim)
        so = np.zeros(self.nhdim)
        """
        os = np.zeros(self.nhdim)
        oo = np.zeros(self.nhdim)
        al = np.zeros(self.nhdim)
        """
        rbz = np.zeros(self.nhdim)
        
        for i in range(self.nhdim):
            rbz[i] = self.bz*i
            ss[i] = adhoc1*sisi[i]/(4*self.pi*rbz[i]**2)#/self.niter #Add cofactors and normalize
            so[i] = adhoc3*sio[i]/(4*self.pi*rbz[i]**2)#/self.niter
            
            """
            os[i] = adhoc7*osi[i]/(4*self.pi*rbz[i]**2)#/self.niter
            oo[i] = adhoc9*oo[i]/(4*self.pi*rbz[i]**2)#/self.niter
            al[i] = adhocc*aa[i]/(4*self.pi*rbz[i]**2)#/self.niter
            """
        df = pd.DataFrame(np.array([distance, ss, so]).T, columns = ["distance","SiSi", "SiO"])
        
        df.to_csv(csv_path)
    








