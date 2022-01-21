# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 11:54:13 2022

@author: kajah
"""
import pandas as pd

class Analyze():
    
    def __init__(self, df):
        self.df = df
    
    def get_value(self, value, tol = 1e-6):
        res = pd.DataFrame(columns = ["Bond", "Label"])
        col_res = []
        label_res = []
        for col in ["dr, Si-Si","dr, Si-O","dr, O-O"]:
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
    str_list = bond.split(", ")
    idx1 = (str_list[0])[1:]
    idx2 = (str_list[1])[:-1]
    return idx1, idx2

def make_2bond(i1, i2, i3):
    new_bond = "(" + i1 + ", " + i2 + ", " + i3 + ")"
    return new_bond

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
    index_col = bond1.columns[0]
    dr_col = bond1.columns[1]
    
    index_col2 = bond2.columns[0]
    dr_col2 = bond2.columns[1]
    
    bonds = []
    bond_length1 = []
    bond_length2 = []
    
    for b1 in bond1[index_col]: # for every index in df1
        i1, i2 = split_idx(b1)
        dr12 = bond1[dr_col].loc[bond1[index_col] == b1] # find bond length corresponding to index b1
        
        for b2 in bond2[index_col]: # for every index in df2
            i3, i4 = split_idx(b2)
            dr34 = bond2[dr_col2].loc[bond2[index_col2] == b2] # find bond length corresponding to index b2
            
            #Check if the indexes overlap:
            if i2 == i3:
                bonds.append(make_2bond(i1, i2, i4))
                bond_length1.append(dr12)
                bond_length2.append(dr34)
                
            elif i1 == i4:
                bonds.append(make_2bond(i3, i1, i2))
                bond_length1.append(dr34)
                bond_length2.append(dr12)
                
            elif i2 == i4:
                bonds.append(make_2bond(i1, i2, i3))
                bond_length1.append(dr12)
                bond_length2.append(dr34)
                
            elif i1 == i3:
                bonds.append(make_2bond(i4, i1, i2))
                bond_length1.append(dr34)
                bond_length2.append(dr12)
    
    new_name = [new_bond_name]*len(bonds)
    
    df = pd.DataFrame(list(zip(bonds, bond_length1, bond_length2, new_name)), columns = ["Index", "Bond lenght 1", "Bond length 2", "Bond name"])
    
    return df
        

        
        

        