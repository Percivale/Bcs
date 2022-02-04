# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 11:54:13 2022

@author: kajah
"""
import pandas as pd
import numpy as np

class Analyze():
    
    def __init__(self, df):
        self.df = df
    
    def get_value(self, value, tol = 1e-6):
        res = pd.DataFrame(columns = ["Bond", "Label"])
        col_res = []
        label_res = []
        for col in self.df.columns:
            #print(self.df[col].loc[abs(self.df[col] - value) <= tol])
            df_res = self.df.loc[abs(self.df[col] - value) <= tol]
            if len(df_res.index):
                col_res.append(col[4:])
                #print(col[4:] + " label")
                #print((col[4:]+" label") == "Si-Si label")
                #print(df_res.columns)
                #label_res.append(df_res["Si-Si label"])
                label_res.append(df_res[col[4:] + " label"].item())
                
        res["Bond"] = col_res
        res["Label"] = label_res
        return res
    
def split_idx(bond):
    str_list = str(bond).split(", ")
    idx1 = (str_list[0])[1:]
    idx2 = (str_list[1])[:-1]
    return idx1, idx2

def make_2bond(i1, i2, i3):
    new_bond = "(" + i1 + ", " + i2 + ", " + i3 + ")"
    return new_bond

def get_dist(i1, i2, xyz_path = "C:\\Users\\kajah\\ntnu\\bachelor_oppg\\xyz_files\\etrap_structures.tar"):
    coord = pd.read_table(xyz_path, skiprows=2, delim_whitespace=True, nrows = 113, names=['atom', 'x', 'y', 'z'])
    #print(coord.index)
    xyz1 = coord.loc[int(i1)]
    xyz2 = coord.loc[int(i2)]
    dr = np.sqrt((xyz1["x"]-xyz2["x"])**2 + (xyz1["y"]-xyz2["y"])**2 + (xyz1["z"]-xyz2["z"])**2)
    return dr
    

def get_2bond(bond1, bond2, new_bond_name):
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
        #print(name1)
        n1, n2 = split_idx(name1)
        #print(n1, n2)
        
        for b2 in bond2[index_col2]: # for every index in df2
            if b1 == b2:
                continue
            
            else:
                i3, i4 = split_idx(b2)
                dr34 = bond2[dr_col2].loc[bond2[index_col2] == b2].item() # find bond length corresponding to index b2
                name2 = bond2[name_col2].loc[bond2[index_col2] == b2].item()
                n3, n4 = split_idx(name2)
                #Check if the indexes overlap:
#                print(n1, n2, n3, n4)
                
                if i2 == i3 and new_bond_name == n1 + "-" + n2 + "-" + n4: #Checks that an index matches, and that the sequence is right
                    #print("Looking for bond: ", new_bond_name,".... \nFound bond: ", n1 + "-" + n2 + "-" + n4)
                    #print("\n\n")
                    bonds.append(make_2bond(i1, i2, i4))
                    bond_length1.append(dr12)
                    bond_length2.append(dr34)
                    length3.append(get_dist(i1, i4))
                    
                elif i1 == i4 and new_bond_name == n3 + "-" + n1 + "-" + n2:
                    #print("Looking for bond: ", new_bond_name,".... \nFound bond: ", n3 + "-" + n1 + "-" + n2)
                    #print("\n\n")
                    bonds.append(make_2bond(i3, i1, i2))
                    bond_length1.append(dr34)
                    bond_length2.append(dr12)
                    length3.append(get_dist(i3, i2))
                    
                elif i2 == i4 and new_bond_name == n1 + "-" + n2 + "-" + n3:
                    #print("Looking for bond: ", new_bond_name,".... \nFound bond: ", n1 + "-" + n2 + "-" + n3)
                    #print("\n\n")
                    bonds.append(make_2bond(i1, i2, i3))
                    bond_length1.append(dr12)
                    bond_length2.append(dr34)
                    length3.append(get_dist(i1, i3))
                    
                elif i1 == i3 and new_bond_name == n4 + "-" + n1 + "-" + n2:
                    #print("Looking for bond: ", new_bond_name,".... \nFound bond: ", n4 + "-" + n1 + "-" + n2)
                    #print("\n\n")
                    bonds.append(make_2bond(i4, i1, i2))
                    bond_length1.append(dr34)
                    bond_length2.append(dr12)
                    length3.append(get_dist(i4, i2))
    
    new_name = [new_bond_name]*len(bonds)
    
    df = pd.DataFrame(list(zip(bonds, bond_length1, bond_length2, length3, new_name)), columns = ["Index", "Bond length 1", "Bond length 2", "Length 3", "Bond name"])
    df = df.drop_duplicates()
    df = df.reset_index(drop = True)
    return df

def get_bond_angle(df, col_a, col_b, col_c):
    a = np.array(df[col_a])
    b = np.array(df[col_b])
    c = np.array(df[col_c])
    
    angle = np.arccos((a**2 + b**2 - c**2)/(2*a*b))
    
    df["Bond Angle"] = angle
    
    return df

        

        
        

        