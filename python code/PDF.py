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
    
    def __init__(self, bz = 0, max_r = 100):
        
        #Set initial values
        self.nmax = 10000
        self.nhdim = 930
        
        self.au = 0.529177              # Bohr radius
        self.boxx = 14.949              # Box size
        self.boxy = 14.949 
        self.boxz = 14.949
        self.beta = 90                  # Box angle (cubic)
        self.pi = np.arccos(-1)         # np.pi ?
        #bz = 0.01                      # Bin size for PDFs (this was too narrow?)
        if not bz:
            self.bz = 0.05
        else:
            self.bz = bz
            
        self.max_r = max_r
            
        #self.niter = 1                  # Number of frames in the xyz file
        self.rcc = self.boxx/2                  #max. distance = 0.5*box
    
    
    def read_xyz(self, inputfile):
        """
        Slightly unnecessary function for retrieving an xyz file

        Parameters
        ----------
        inputfile : String
            File path to file.

        Returns
        -------
        data : DataFrame
            Tabe containing data from the given xyz file.
        """
        
        data = pd.read_table(inputfile, skiprows=2, delim_whitespace=True, nrows = 216, names=['atom', 'x', 'y', 'z'])
        return data
    
    def get_dist(self, i1, i2, xyz_path, vec = False):
        """
        Calculates distance between atom 1 and atom 2 taking into account the box.

        Parameters
        ----------
        i1 : string
            Index of atom 1 in xyz file.
        i2 : string
            Index of atom 2 in xyz file.
        xyz_path : string, optional
            Pathway to xyz file that contain the coordinates. The default is "C:\\Users\\kajah\\ntnu\\bachelor_oppg\\xyz_files\\etrap_structures.tar".

        Returns
        -------
        dr : float
            Cartesian distance between atom 1 and atom 2.

        """
        
        coord = self.read_xyz(xyz_path)
        #print(coord.index)
        xyz1 = coord.loc[int(i1)]
        xyz2 = coord.loc[int(i2)]
        
        x1 = xyz1["x"]#.values
        y1 = xyz1["y"]#.values
        z1 = xyz1["z"]#.values
        x1 = x1-self.boxx*np.rint(x1/self.boxx)
        y1 = y1-self.boxy*np.rint(y1/self.boxy)
        z1 = z1-self.boxz*np.rint(z1/self.boxz)
        
        x2 = xyz2["x"]#.values
        y2 = xyz2["y"]#.values
        z2 = xyz2["z"]#.values
        x2 = x2-self.boxx*np.rint(x2/self.boxx)
        y2 = y2-self.boxy*np.rint(y2/self.boxy)
        z2 = z2-self.boxz*np.rint(z2/self.boxz)
        
        dx = x1 - x2 - self.boxx*np.rint((x1-x2)/self.boxx)
        dy = y1 - y2 - self.boxy*np.rint((y1-y2)/self.boxy)
        dz = z1 - z2 - self.boxz*np.rint((z1-z2)/self.boxz)
        
        #dr = np.array([dx, dy, dz])
        
        if not vec:
            dr = np.sqrt(dx**2 + dy**2 + dz**2)
        else:
            dr = np.array([dx, dy, dz])
            
        return dr
    

    def calculate_pdf(self, xyz_path, csv_path = ""):
        """
        Calculates pdf values from xyz file, and can write the results to a csv file.

        Parameters
        ----------
        xyz_path : string
            Path to xyz file.
        csv_path : string, optional
            Path where csv file should be written to. The default is "" and nothing will be written to file. 

        Returns
        -------
        df : DataFrame
            Contains bond lengths, indexes, atom names and pdf values.

        """
        
        a_sio2 = self.read_xyz(xyz_path)
        
        atoms = a_sio2["atom"].value_counts()
        
        nSi = atoms["Si"]     # nr of Si
        nO = atoms["O"]       # nr of O 
        nions = nSi + nO      # nr of atoms
        print("nr of Si: ", nSi)
        print("nr of O: ", nO)
        

        sp = a_sio2["atom"].values
        x = a_sio2["x"].values
        y = a_sio2["y"].values
        z = a_sio2["z"].values
        
        #idx = a_sio2.index
        
        adhoc1 = self.boxx*self.boxy*self.boxz/(nSi*(nSi-1))/self.bz            #Cofactors for each PDF
        adhoc3 = self.boxx*self.boxy*self.boxz/(nSi*nO - (nSi + nO))/self.bz
        #adhoc7 = self.boxx*self.boxy*self.boxz/(nO*nSi - (nO + nSi))/self.bz
        adhoc9 = self.boxx*self.boxy*self.boxz/(nO*(nO-1))/self.bz
        #adhocc = self.boxx*self.boxy*self.boxz/(nions*(nions-1))/self.bz

        x = x-self.boxx*np.rint(x/self.boxx)
        y = y-self.boxy*np.rint(y/self.boxy)
        z = z-self.boxz*np.rint(z/self.boxz)

        sisi = np.zeros(self.nhdim)      #Si-Si bonds?
        oo = np.zeros(self.nhdim)        #O-O bonds?
        sio = np.zeros(self.nhdim)       #Si-O bonds?
        #osi = np.zeros(self.nhdim)       #O-Si bonds? Why does the sequence matter?
        #aa = np.zeros(self.nhdim)        # no idea
        
        distance = np.zeros(self.nhdim)
        distancesio = np.zeros(self.nhdim)
        distoo = np.zeros(self.nhdim)
        isisi = np.zeros(self.nhdim, dtype = object)
        isio = np.zeros(self.nhdim, dtype = object)
        ioo = np.zeros(self.nhdim, dtype = object)
        atom_names_ss = np.zeros(self.nhdim, dtype = object)
        atom_names_so = np.zeros(self.nhdim, dtype = object)
        atom_names_oo = np.zeros(self.nhdim, dtype = object)
        
        
        #distancesio = []
        #sio = []
        #isio = []
        #atom_names_so = []
        
        count = 0
        for i in range(nions):
            for j in range(nions):
                dx = x[i] - x[j] - self.boxx*np.rint((x[i]-x[j])/self.boxx)
                dy = y[i] - y[j] - self.boxy*np.rint((y[i]-y[j])/self.boxy)
                dz = z[i] - z[j] - self.boxz*np.rint((z[i]-z[j])/self.boxz)
                
                dr = np.sqrt(dx**2 + dy**2 + dz**2)
                
                
                #Si
                if dr <= self.rcc and dr <= self.max_r:
                    ll = int(np.rint(dr/self.bz))
                    #print(ll)
                    
                    if sp[i] == "Si" and sp[j] == "Si" and i != j:
                        sisi[ll] = sisi[ll] + 1
                        distance[ll] = dr
                        isisi[ll] = str(i) + " " + str(j)
                        atom_names_ss[ll] = sp[i] + " " + sp[j]
                        
                        
                    elif sp[i] == "Si" and sp[j] == "O":
                        sio[ll] = sio[ll] + 1
                        distancesio[ll] = dr
                        isio[ll] = str(i) + " " + str(j)
                        atom_names_so[ll] = sp[i] + " " + sp[j]
                        count += 1

                        
                        
                    elif sp[i] == "O" and sp[j] == "O" and i != j:
                        oo[ll] = oo[ll] + 1
                        distoo[ll] = dr
                        ioo[ll] =  str(i) + " " + str(j)
                        atom_names_oo[ll] = sp[i] + " " + sp[j]
                    
        print("Number of bonds evaluated: ", count)
        #---------------------------------------------------------------------
        ss = np.zeros(self.nhdim)
        so = np.zeros(self.nhdim)
        oo_ = np.zeros(self.nhdim)
        """
        os = np.zeros(self.nhdim)
        al = np.zeros(self.nhdim)
        """
        rbz = np.zeros(self.nhdim)
        
        for i in range(self.nhdim):
            rbz[i] = self.bz*i
            ss[i] = adhoc1*sisi[i]/(4*self.pi*rbz[i]**2)#/self.niter #Add cofactors and normalize
            so[i] = adhoc3*sio[i]/(4*self.pi*rbz[i]**2)#/self.niter
            oo_[i] = adhoc9*oo[i]/(4*self.pi*rbz[i]**2)#/self.niter
            
            """
            os[i] = adhoc7*osi[i]/(4*self.pi*rbz[i]**2)#/self.niter
            
            al[i] = adhocc*aa[i]/(4*self.pi*rbz[i]**2)#/self.niter
            """
        df = pd.DataFrame(np.array([distance, distancesio, distoo, ss, so, oo_, isisi, isio, ioo, atom_names_ss, atom_names_so, atom_names_oo]).T, 
                          columns = ["dr, Si-Si","dr, Si-O","dr, O-O", "Si-Si", "Si-O", "O-O", "Si-Si label", "Si-O label", "O-O label", "atoms_Si-Si", "atoms_Si-O", "atoms_O-O"])
        df = df.loc[(df != 0).any(axis=1)]
        df = df.dropna()
        
        if csv_path != "":
            df.to_csv(csv_path)
        else:
            return df

    def get_bonds(self, xyz_path, max_r_sio, max_r_sisi, max_r_oo, csv_path = ""):
        """
        Calculates pdf values from xyz file, and can write the results to a csv file.

        Parameters
        ----------
        xyz_path : string
            Path to xyz file.
        csv_path : string, optional
            Path where csv file should be written to. The default is "" and nothing will be written to file. 

        Returns
        -------
        df : DataFrame
            Contains bond lengths, indexes, atom names and pdf values.

        """
        
        a_sio2 = self.read_xyz(xyz_path)
        
        atoms = a_sio2["atom"].value_counts()
        
        nSi = atoms["Si"]     # nr of Si
        nO = atoms["O"]       # nr of O 
        nions = nSi + nO      # nr of atoms
        

        sp = a_sio2["atom"].values
        x = a_sio2["x"].values
        y = a_sio2["y"].values
        z = a_sio2["z"].values
        

        x = x-self.boxx*np.rint(x/self.boxx)
        y = y-self.boxy*np.rint(y/self.boxy)
        z = z-self.boxz*np.rint(z/self.boxz)
        
        distance = []
        index = []
        name = []
        
        for i in range(nions):
            for j in range(nions):
                dx = x[i] - x[j] - self.boxx*np.rint((x[i]-x[j])/self.boxx)
                dy = y[i] - y[j] - self.boxy*np.rint((y[i]-y[j])/self.boxy)
                dz = z[i] - z[j] - self.boxz*np.rint((z[i]-z[j])/self.boxz)
                
                dr = np.sqrt(dx**2 + dy**2 + dz**2)
                
                if dr <= self.rcc and i != j:  
                    if (sp[i] + " " + sp[j] == "Si O" or sp[i] + " " + sp[j] == "O Si") and dr <= max_r_sio: 
                        distance.append(dr)
                        index.append(str(i) + " " + str(j))
                        name.append(sp[i] + " " + sp[j])
                        
                    elif sp[i] + " " + sp[j] == "Si Si" and dr <= max_r_sisi:
                        distance.append(dr)
                        index.append(str(i) + " " + str(j))
                        name.append(sp[i] + " " + sp[j])
                    elif sp[i] + " " + sp[j] == "O O" and dr <= max_r_oo:
                        distance.append(dr)
                        index.append(str(i) + " " + str(j))
                        name.append(sp[i] + " " + sp[j])
                    

        df = pd.DataFrame(np.array([distance, index, name]).T, columns = ["dr", "index", "name"])
        df = df.loc[(df != 0).any(axis=1)]
        df = df.dropna()
        df = df.reset_index(drop=True)
        
        if csv_path != "":
            df.to_csv(csv_path)
        else:
            return df

def get_value(df, value, tol = 1e-6):
    """
    Finds all bond lengths that have a specific distance (with a tolerance), and returns the label.

    Parameters
    ----------
    df : DataFrame
        Contains bond lengths, labels.
    value : float
        Numerical value that should be found.
    tol : float, optional
        Tolerance for value. The default is 1e-6.

    Returns
    -------
    res : DataFrame
        Contains all rows that contain value.

    """
    
    res = pd.DataFrame(columns = ["Bond", "Label"])
    col_res = []
    label_res = []
    for col in df.columns:
        df_res = df.loc[abs(df[col] - value) <= tol]
        if len(df_res.index):
            col_res.append(col[4:])
            label_res.append(df_res[col[4:] + " label"].item())
            
    res["Bond"] = col_res
    res["Label"] = label_res
    return res
    
def split_idx(bond, delimiter = " "):
    """
    Splits the index and returns the numbers

    Parameters
    ----------
    bond : string
        Of the form "(int, int)". Is two atom's indexes.

    Returns
    -------
    idx1 : string
        Index of atom 1.
    idx2 : string
        Index of atom 2.

    """
    
    str_list = str(bond).split(delimiter)
    #print(str_list)
    idx1 = (str_list[0])
    idx2 = (str_list[1])
    
    if len(str_list)>2:
        idx3 = (str_list[2])
        return idx1, idx2, idx3
    else:
        return idx1, idx2


def make_2bond(i1, i2, i3):
    """
    Formats an index for a system with three atoms (two bonds)

    Parameters
    ----------
    i1 : string
        Index of atom 1.
    i2 : string
        Index of atom 2.
    i3 : string
        Index of atom 3.

    Returns
    -------
    new_bond : string
        New index for system with two bonds. 
    """
    
    new_bond = i1 + " " + i2 + " " + i3
    return new_bond


    

def get_2bond(bond1, bond2, new_bond_name, pdf_system, xyz_path):
    """
    Takes in two dataframes. Checks if the index (i, j) overlap. If it does, the bonds appear in 
    the new dataframe.

    Parameters
    ----------
    bond1 : DataFrame
        Columns: bond length, index.
    bond2 : DataFrame
        Columns: bond length, index.
    new_bond_name: string.
        New name for the bonds: Si-O-Si
        
    Returns
    -------
    df : DataFrame
        Contains two bonds in each row. 
    
        bond-length 1 | bond-length 2| index     | Bond type
        dr12          |dr23          | (i, j, k) | Si-O-Si
        ...

    """
    index_col = bond1.columns[1]
    dr_col = bond1.columns[0]
    name_col = bond1.columns[2]
    
    index_col2 = bond2.columns[1]
    dr_col2 = bond2.columns[0]
    name_col2 = bond2.columns[2]
    
    bonds = []
    bond_length1 = []
    bond_length2 = []
    length3  =[]
    
    for b1 in bond1[index_col]: # for every index in df1
        i1, i2 = split_idx(b1)
        dr12 = bond1[dr_col].loc[bond1[index_col] == b1].item() # find bond length corresponding to index b1
        name1 = bond1[name_col].loc[bond1[index_col] == b1].item()
        n1, n2 = split_idx(name1)
        
        for b2 in bond2[index_col2]: # for every index in df2
            if b1 == b2: # Cannot be bonds between the same atom, so skip if it's the same index
                continue
        
            else:
                i3, i4 = split_idx(b2)
                dr34 = bond2[dr_col2].loc[bond2[index_col2] == b2].item() # find bond length corresponding to index b2
                name2 = bond2[name_col2].loc[bond2[index_col2] == b2].item()
                n3, n4 = split_idx(name2)
                #Check if the indexes overlap:
                #print("Name: ", n1, n2, n3, n4)
                
                if i2 == i3 and new_bond_name == n1 + " " + n2 + " " + n4: #Checks that an index matches, and that the sequence is right
                    #print("Looking for bond: ", new_bond_name,".... \nFound bond: ", n1 + "-" + n2 + "-" + n4)
                    #print("\n\n")
                    bonds.append(make_2bond(i1, i2, i4))
                    bond_length1.append(dr12)
                    bond_length2.append(dr34)
                    length3.append(pdf_system.get_dist(i1, i4, xyz_path))
                    
                elif i1 == i4 and new_bond_name == n3 + " " + n1 + " " + n2:
                    #("Looking for bond: ", new_bond_name,".... \nFound bond: ", n3 + "-" + n1 + "-" + n2)
                    #print("\n\n")
                    bonds.append(make_2bond(i3, i1, i2))
                    bond_length1.append(dr34)
                    bond_length2.append(dr12)
                    length3.append(pdf_system.get_dist(i3, i2, xyz_path))
                    
                elif i2 == i4 and new_bond_name == n1 + " " + n2 + " " + n3:
                    #print("Looking for bond: ", new_bond_name,".... \nFound bond: ", n1 + "-" + n2 + "-" + n3)
                    #print("\n\n")
                    bonds.append(make_2bond(i1, i2, i3))
                    bond_length1.append(dr12)
                    bond_length2.append(dr34)
                    length3.append(pdf_system.get_dist(i1, i3, xyz_path))
                    
                elif i1 == i3 and new_bond_name == n4 + " " + n1 + " " + n2:
                    #print("Looking for bond: ", new_bond_name,".... \nFound bond: ", n4 + "-" + n1 + "-" + n2)
                    #print("\n\n")
                    bonds.append(make_2bond(i4, i1, i2))
                    bond_length1.append(dr34)
                    bond_length2.append(dr12)
                    length3.append(pdf_system.get_dist(i4, i2, xyz_path))
                    
                
    
    new_name = [new_bond_name]*len(bonds)
    
    df = pd.DataFrame(list(zip(bonds, bond_length1, bond_length2, length3, new_name)), columns = ["Index", "Bond length 1", "Bond length 2", "Length 3", "Bond name"])
    df = df.loc[(df["Length 3"] != 0)]
    df = df.reset_index(drop=True)
    return df




def get_bond_angle(df, col_a, col_b, col_c):
    """
    Calculates the bond angle from the distance between all three atoms.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    col_a : TYPE
        DESCRIPTION.
    col_b : TYPE
        DESCRIPTION.
    col_c : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    
    a = np.array(df[col_a], dtype = float)
    b = np.array(df[col_b], dtype = float)
    c = np.array(df[col_c], dtype = float)
    
    angle = np.rad2deg(np.arccos((a**2 + b**2 - c**2)/(2*a*b))) # Cosine law ##
    
    df["Bond Angle"] = angle
    
    return df

def calc_dihedral(dr12, dr23, dr34):
    #top = np.vdot(np.cross(dr12, dr23), np.cross(dr23, dr34))
    #bot = np.linalg.norm(np.cross(dr12, dr23))*np.linalg.norm(np.cross(dr23, dr34))
    #angle = np.arccos(top/bot)
    
    #x = np.vdot(dr23, np.cross(np.cross(dr12, dr23), np.cross(dr23, dr34)))
    #y = np.linalg.norm(dr23)*np.vdot(np.cross(dr12, dr23), np.cross(dr23, dr34))
    
    x = np.vdot(np.linalg.norm(dr23)*dr12, np.cross(dr23, dr34))
    y = np.vdot(np.cross(dr12, dr23), np.cross(dr23, dr34))
    
    angle = np.arctan2(x, y)
    return -np.rad2deg(angle)



def get_vec(xyz_df, idx1, idx2):
    print(xyz_df.index)
    r1 = xyz_df[["x", "y", "z"]].loc[int(idx1)].values
    r2 = xyz_df[["x", "y", "z"]].loc[int(idx2)].values
    dr12 = r2-r1
    
    return dr12

def get_dihedrals(df, df_2, xyz_path, system):
  
    idx2_col = df_2.columns[0]
    name2_col = df_2.columns[-1]

    idx_col = df.columns[1]
    name_col = df.columns[2]

    
    index = []
    bl1 = []
    bl2 = []
    bl3 = []
    angle = []
    name = []

    for chain in df_2[idx2_col]:

        idx1, idx2, idx3 = split_idx(chain)

        dr12 = system.get_dist(idx1, idx2, xyz_path, vec = True)
        dr23 = system.get_dist(idx2, idx3, xyz_path, vec = True)
        
        n1, n2, n3 = split_idx(df_2[name2_col].loc[df_2[idx2_col] == chain].item())
        
        for bond in df[idx_col]:
            idx4, idx5 = split_idx(bond)
            
            #dr = get_vec(xyz_df, idx4, idx5)
            dr = system.get_dist(idx4, idx5, xyz_path, vec = True)
            n4, n5 = split_idx(df[name_col].loc[df[idx_col] == bond].item())
            #dr = df[dr_col].loc[df[idx_col] == bond].item()
            
            #NEED TO CHECK IF CIS OR TRANS?
            if idx4 == idx3 and idx5 != idx2 and idx5 != idx1:
                #print("Dihedral angle found: ", chain + " " + idx5)
                #print("b1: ", dr12)
                #print("b2: ", dr23)
                #print("b3: ", dr)
                
                index.append(chain + " " + idx5)
                bl1.append(np.linalg.norm(dr12))
                bl2.append(np.linalg.norm(dr23))
                bl3.append(np.linalg.norm(dr))
                angle.append(calc_dihedral(dr12, dr23, dr))
                name.append(n1 + " " + n2 + " " + n3 + " " + n5)

            elif idx4 == idx1 and idx5 != idx3 and idx5 != idx2:
                #print("Dihedral angle found: ", idx5 + " " + chain)
                #print("b1: ", dr)
                #print("b2: ", dr12)
                #print("b3: ", dr23)
                index.append(idx5 + " " + chain)
                bl1.append(np.linalg.norm(dr))
                bl2.append(np.linalg.norm(dr12))
                bl3.append(np.linalg.norm(dr23))
                angle.append(calc_dihedral(-dr, dr12, dr23))
                name.append(n5 + " " + n1 + " " + n2 + " " + n3)

            elif idx5 == idx1 and idx4 != idx3 and idx4 != idx2:
                #print("Dihedral angle found: ", idx4 + " " + chain)
                #print("b1: ", dr)
                #print("b2: ", dr12)
                #print("b3: ", dr23)
                index.append(idx4 + " " +chain)
                bl1.append(np.linalg.norm(dr))
                bl2.append(np.linalg.norm(dr12))
                bl3.append(np.linalg.norm(dr23))
                angle.append(calc_dihedral(dr, dr12, dr23))
                name.append(n4 + " " + n1 + " " + n2 + " " + n3)

            elif idx5 == idx3 and idx4 != idx1 and idx4 != idx2:
                #print("Dihedral angle found: ", chain + " " + idx4)
                #print("b1: ", dr12)
                #print("b2: ", dr23)
                #print("b3: ", dr)
                index.append(chain + " " + idx4)
                bl1.append(np.linalg.norm(dr12))
                bl2.append(np.linalg.norm(dr23))
                bl3.append(np.linalg.norm(dr))
                angle.append(calc_dihedral(dr12, dr23, -dr))
                name.append(n1 + " " + n2 + " " + n3 + " " + n4)
                
    dihedral_df = pd.DataFrame(list(zip(index, bl1, bl2, bl3, angle, name)), columns = ["Index", "Bond length 1", "Bond length 2", "Bond length 3", "Dihedral angle","Bond name"])
    dihedral_df = dihedral_df.drop_duplicates()
    dihedral_df = remove_rev_dupes(dihedral_df)
    dihedral_df = dihedral_df.reset_index(drop=True)
    return dihedral_df
    
def remove_rev_dupes(df):
    keep = []
    for i in df["Index"]:
        for j in df["Index"]:
            if i == " ".join(j.split(" ")[::-1]) and j not in keep:
                keep.append(i)
                df = df.drop(df.index[df["Index"] == j])
    return df

def construct_df(xyz_path, max_r = 100):
    bz = 0.008
    etrap = PDF(bz, max_r)
    #df = etrap.calculate_pdf(xyz_path, csv_path = "")
    df = etrap.get_bonds(xyz_path)
    
    sio_bonds = df.loc[df["name"] == "Si O"]
    #sio_bonds.to_csv("sio.csv")
    
    #sio_bonds = sio_bonds.append(df.loc[df["name"] == "O Si"])
    #sio_bonds = df[["dr, Si-O", "Si-O label", "atoms_Si-O"]]
    #sio_bonds = sio_bonds.loc[(sio_bonds["dr, Si-O"] != 0)]
   
    #sisi_bonds = df[["dr, Si-Si", "Si-Si label", "atoms_Si-Si"]]
    #sisi_bonds = sisi_bonds.loc[(sisi_bonds["dr, Si-Si"] != 0)]

    #oo_bonds = df[["dr, O-O", "O-O label", "atoms_O-O"]]
    #oo_bonds = oo_bonds.loc[(oo_bonds["dr, O-O"] != 0)]


    siosi = get_2bond(sio_bonds, sio_bonds, "Si O Si", etrap)
    #sisio = get_2bond(sisi_bonds, sio_bonds, "Si-Si-O", etrap)
    #sisisi = get_2bond(sisi_bonds, sisi_bonds, "Si-Si-Si", etrap)
    #sioo = get_2bond(sio_bonds, oo_bonds, "Si-O-O", etrap)
    osio = get_2bond(sio_bonds, sio_bonds, "O Si O", etrap)
    
    
    df = pd.DataFrame().append([siosi, osio]).drop_duplicates(["Bond length 1", "Bond length 2", "Length 3"], ignore_index= True)
    
    df = remove_rev_dupes(df)
    """
    keep = []     
    
    for i in df["Index"]:
        for j in df["Index"]:
            if i == " ".join(j.split(" ")[::-1]) and j not in keep:
                keep.append(i)
                df = df.drop(df.index[df["Index"] == j])
    
    """    
    
    #df = pd.DataFrame().append([siosi, sisio, sisisi, sioo, osio]).drop_duplicates(["Bond length 1", "Bond length 2", "Length 3"], ignore_index= True)
    
    df = get_bond_angle(df, "Bond length 1", "Bond length 2", "Length 3")
    
    return df

def get_ML_data(xyz_path, csv_path):
    """
    - a routine that prints out the atomic indices  and distances for Si-O, Si-Si, O-O by using cutoffs 2.0, 3.5 and 3.0 Å, respectively. 
    - a routine that prints out the atom indices and angles for O-Si-O and Si-O-Si by using a cutoff 2.0 Å
    - a routine that prints out the atom indices and dihedral angles O-Si-O-Si by using a cutoff 2.0 Å

    Returns
    -------
    None.

    """
    bz = 0.008
    etrap = PDF(bz)
    df = etrap.get_bonds(xyz_path, max_r_sio = 2.0, max_r_sisi = 3.5, max_r_oo=3.0)
    #df = remove_rev_dupes(df)
    
    sio_bonds = df.loc[df["name"] == "Si O"]
    sio_bonds = sio_bonds.append(df.loc[df["name"] == "O Si"])
    #sisi_bonds = df.loc[df["name"] == "Si Si"]
    #oo_bonds = df.loc[df["name"] == "O O"]
    
    #print("Si-O bonds with cutoff 2.0: \n---------------------------------------\n")
    #print(sio_bonds)
    #print("Si-Si bonds with cutoff 3.5: \n---------------------------------------\n")
    #print(sisi_bonds)
    #print("O-O bonds with cutoff 3.0: \n---------------------------------------\n")
    #print(oo_bonds)
    
    df.to_csv(csv_path + "single_bonds.csv")
    

    #sio_bonds has cutoff 2.0Å, so siosi will also have this cutoff distance. 
    siosi = get_2bond(sio_bonds, sio_bonds, "Si O Si", etrap, xyz_path)
    osio = get_2bond(sio_bonds, sio_bonds, "O Si O", etrap, xyz_path)
    #print("Si-O-Si bonds with cutoff 2.0Å: \n----------------------------------------------\n")
    #print(siosi)
    #print("O-Si-O bonds with cutoff 2.0Å: \n----------------------------------------------\n")
    #print(osio)
    
    df_2 = pd.DataFrame().append([siosi, osio]).drop_duplicates(["Bond length 1", "Bond length 2", "Length 3"], ignore_index= True)
    df_2 = remove_rev_dupes(df_2)
    df_2 = get_bond_angle(df_2, "Bond length 1", "Bond length 2", "Length 3") 
    df_2.to_csv(csv_path + "angles.csv")
    

    
    df_dihedral = get_dihedrals(sio_bonds, df_2[["Index", "Bond length 1", "Bond length 2", "Length 3", "Bond name"]], xyz_path, etrap)
    #df_dihedral = remove_rev_dupes(df_dihedral)
    df_dihedral.to_csv(csv_path + "dihedral_angles.csv")
    
    
    

    








