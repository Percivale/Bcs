# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 14:40:34 2022

@author: kajah
"""
import numpy as np
import pandas as pd


class AmorphousOxide:
    def __init__(self, nr_atoms, nr_si, nr_o, boxsize, xyz_path):
        self.nrAtoms = nr_atoms
        self.nrSi = nr_si
        self.nrO = nr_o
        self.boxX = boxsize[0]
        self.boxY = boxsize[1]
        self.boxZ = boxsize[2]
        
        xyz_df = pd.read_table(xyz_path, skiprows=2, delim_whitespace=True, nrows = self.nrAtoms, names=['atom', 'x', 'y', 'z'])
        
        #Retrieving data from xyz file and putting them in arrays:
        self.nameArray = list(xyz_df["atom"].to_numpy(dtype = str))
        
        x = xyz_df["x"].to_numpy()
        y = xyz_df["y"].to_numpy()
        z = xyz_df["z"].to_numpy()
        
        #Implementing periodic boundaries:
        self.xArray = x - self.boxX*np.rint(x/self.boxX)
        self.yArray = y - self.boxY*np.rint(y/self.boxY)
        self.zArray = z - self.boxZ*np.rint(z/self.boxZ)
        
    def get_chain2(self, cutoffs):
        index, names, bond_lengths, dx, dy, dz = make_bonds(self.xArray, self.yArray, self.zArray, self.nameArray, 
                                                self.boxX, self.boxY, self.boxZ, cutoffs)
                                                #make_chain(2, self.xArray, self.yArray, self.zArray, self.nameArray, 
                                                #self.boxX, self.boxY, self.boxZ, cutoffs)
        #print(bond_vec)
        df = pd.DataFrame(np.array([index, names, bond_lengths, dx, dy, dz]).T, columns = ["Index", "Name", "Bond length", "dx", "dy", "dz"])
        return df
    
    def get_dihedral_angles(self, cutoffs):
        index2, names2, bond_lengths2, dx2, dy2, dz2 = make_bonds(self.xArray, self.yArray, self.zArray, self.nameArray, 
                                                self.boxX, self.boxY, self.boxZ, cutoffs)

        index1, names1, bl1, bl2, bl3, dx1, dy1, dz1, dx12, dy12, dz12 = match_bonds(index2, names2, bond_lengths2, dx2, dy2, dz2, index2, names2, bond_lengths2, dx2, dy2, dz2)
        
        index, names, bl1, bl2, bl3, angles = get_dihedrals(index1, names1, bl1, bl2, dx1, dy1, dz1, dx12, dy12, dz12, index2, names2, bond_lengths2, dx2, dy2, dz2)
        print(len(angles))
        df = pd.DataFrame(np.array([index, names, bl1, bl2, bl3, angles]).T, columns = ["Index", "Name", "Bond length 1", "Bond length 2", "Bond length 3", "Dihedral Angles"])
        return df
    
    def get_chain3_2(self, df1, df2):
        index1 = df1["Index"].to_numpy(dtype = str)
        names1 = df1["Name"].to_numpy(dtype = str)
        bondlengths1 = df1["Bond length"].to_numpy(dtype = float)
        dx1 = df1["dx"].to_numpy(dtype = float)
        dy1 = df1["dy"].to_numpy(dtype = float)
        dz1 = df1["dz"].to_numpy(dtype = float)

        

        index2 =  df2["Index"].to_numpy(dtype = str)
        names2 = df2["Name"].to_numpy(dtype = str)
        bondlengths2 = df2["Bond length"].to_numpy(dtype = float)
        dx2 = df2["dx"].to_numpy(dtype = float)
        dy2 = df2["dy"].to_numpy(dtype = float)
        dz2 = df2["dz"].to_numpy(dtype = float)

        
        index, names, bl1, bl2, bl3, dx1, dy1, dz1, dx2, dy2, dz2 = match_bonds(index1, names1, bondlengths1, dx1, dy1, dz1, index2, names2, bondlengths2, dx2, dy2, dz2)
        
        angle = calc_angle(np.array(bl1), np.array(bl2), np.array(bl3))
        #
        #print(angle)
        df = pd.DataFrame(np.array([index, names, bl1, bl2, angle]).T, columns = ["Index", "Name", "Bond length 1", "Bond length2","Angle"])
        df = df.drop_duplicates()
        #df = remove_rev_dupes(df)
        return df


        
def calc_angle(a, b, c):
    angle = np.rad2deg(np.arccos((a**2 + b**2 - c**2)/(2*a*b))) # Cosine law ##
    return angle    

def remove_rev_dupes(df):
    keep = []
    for i in df["Index"]:
        for j in df["Index"]:
            if i == " ".join(j.split(" ")[::-1]) and j not in keep:
                keep.append(i)
                df = df.drop(df.index[df["Index"] == j])
    return df
    
def make_bonds(x_array, y_array, z_array, name_array, boxx, boxy, boxz, cutoffs):
    cut_sisi = cutoffs[0]
    cut_sio = cutoffs[1]
    cut_oo = cutoffs[2]
    
    names = []
    index = []
    bondlengths = []
    x_12 = []
    y_12 = []
    z_12 = []

    for i in range(len(name_array)):
        x1 = x_array[i]
        y1 = y_array[i]
        z1 = z_array[i]
        
        for j in range(len(name_array)):
            x2 = x_array[j]
            y2 = y_array[j]
            z2 = z_array[j]
            
            #find distance between the atoms (with periodic boundary):
            dx = x1 - x2 - boxx*np.rint((x1-x2)/boxx)
            dy = y1 - y2 - boxy*np.rint((y1-y2)/boxy)
            dz = z1 - z2 - boxz*np.rint((z1-z2)/boxz) 
            dr = np.sqrt(dx**2 + dy**2 + dz**2)
            #print(dr)
        
            
            if (((name_array[i] == "Si" and name_array[j] == "O") or (name_array[i] == "O" or name_array[j] == "Si")) and dr <= cut_sio) or (name_array[i] == "Si" and name_array[j] == "Si" and dr <= cut_sisi) or (name_array[i] == "O" and name_array[j] == "O" and dr <= cut_oo):
                chain_name = [name_array[i], name_array[j]]
                #chain_bondlengths = dr
                chain_index = [str(i), str(j)]
                #chain_vec = [str(dx), str(dy), str(dz)]
               #print(chain_name, chain_index)
                if chain_index[::-1] not in index and dr != 0:
                    names.append(" ".join(chain_name))
                    bondlengths.append(dr)
                    index.append(" ".join(chain_index))
                    x_12.append(dx)
                    y_12.append(dy)
                    z_12.append(dz)
                    
                    #bond_vec.append(" ".join(chain_vec))
                    
    return index, names, bondlengths, x_12, y_12, z_12#, bond_vec


def match_bonds(index1, names1, bondlengths1, dx1, dy1, dz1, index2, names2, bondlengths2, dx2, dy2, dz2):
    bl1 = []
    bl2 = []
    bl3 = []
    names = []
    index = []
    dx_1 =[]
    dy_1 =[]
    dz_1 =[]
    dx_2 =[]
    dy_2 =[]
    dz_2 =[]
    
    for i in range(len(index1)):
        idx1 = index1[i].split(" ")
        i1, i2 = idx1[0], idx1[-1]
        nm1 = names1[i].split(" ")
        
        for j in range(len(index2)):
            idx2 = index2[j].split(" ")
            i3, i4 = idx2[0], idx2[-1]
            nm2 = names2[j].split(" ")

            if i == j or idx1 == idx2[::-1]:
                continue
            
            chain_name = " ".join(nm1) + " " + " ".join(nm2[1:])
            
            if (chain_name == "Si O Si" or chain_name == "O Si O" or chain_name == "Si Si Si") and i2 == i3:
                bl1.append(bondlengths1[i])
                bl2.append(bondlengths2[j])
                length3 = np.linalg.norm(np.array([dx2[j], dy2[j], dz2[j]])+np.array([dx1[i], dy1[i], dz1[i]]))
                bl3.append(length3)
                names.append(chain_name)
                index.append(" ".join(idx1) + " " + " ".join(idx2[1:]))
                
                dx_1.append(dx1[i])
                dy_1.append(dy1[i])
                dz_1.append(dz1[i])
                
                dx_2.append(dx2[j])
                dy_2.append(dy2[j])
                dz_2.append(dz2[j])
    return index, names, bl1, bl2, bl3, dx_1, dy_1, dz_1, dx_2, dy_2, dz_2#, bv1, bv2 

def calc_dihedral(dr12, dr23, dr34):
    x = np.vdot(np.linalg.norm(dr23)*dr12, np.cross(dr23, dr34))
    y = np.vdot(np.cross(dr12, dr23), np.cross(dr23, dr34))
    
    angle = np.arctan2(x, y)
    return -np.rad2deg(angle)

def get_dihedrals(index1, names1, bondlengths1, bondlengths12, dx11, dy11, dz11, dx12, dy12, dz12, index2, names2, bondlengths2, dx2, dy2, dz2):
    index = []
    bl1 = []
    bl2 = []
    bl3 = []
    
    angle = []
    name = []

    for i in range(len(index1)):
        idx1, idx2, idx3 = index1[i].split(" ")
        n1, n2, n3 = names1[i].split(" ")

        dr12 = np.array([dx11[i], dy11[i], dz11[i]])
        dr23 = np.array([dx12[i], dy12[i], dz12[i]])
        
        for j in range(len(index2)):
            idx4, idx5 = index2[j].split(" ")
            n4, n5 = names2[j].split(" ")
            
            dr34 = np.array([dx2[j], dy2[j], dz2[j]])

            if idx4 == idx3 and idx5 != idx2 and idx5 != idx1 and (names1[i] + " " + n5 == "Si O Si O" or names1[i] + " " + n5 == "O Si O Si"):
                if " ".join([idx1, idx2, idx3, idx5][::-1]) not in index:
                    index.append(index1[i] + " " + idx5)
                    name.append(names1[i] + " " + n5)
                    
                    bl1.append(np.linalg.norm(dr12))
                    bl2.append(np.linalg.norm(dr23))
                    bl3.append(np.linalg.norm(dr34))
                    
                    angle.append(calc_dihedral(dr12, dr23, dr34))
    
    return index, name, bl1, bl2, bl3, angle

def get_rings(index, n):
    rings = []
    dupes = []
    for i in range(len(index)):
        idx1, idx2 = index[i].split(" ")
        for k in range(len(index)):
            idx4, idx3 = index[k].split(" ")
            #print(idx1, idx2, idx3, idx4)
            
            if i == k:
                continue
            else:
                pot_ring1 = idx1 + " " + idx2 + " " + idx4
                pot_ring2 = idx2 + " " + idx4 + " " + idx1
                pot_ring3 = idx4 + " " + idx1 + " " + idx2
                if idx2 == idx3 and (idx1 + " " + idx4 in index): #and pot_ring1 not in rings: #and (pot_ring2 not in rings or pot_ring3 not in rings):
                    #chain.append(idx3)
                    rings.append(pot_ring1)
                    dupes.append(pot_ring2)
                    dupes.append(pot_ring3)
                    #break #start again to look for next atom in chain
    dupes = list(set(dupes))
    print(dupes)
    #for el in dupes:
    #    try:
    #        rings.remove(el)
    #        pass
    return rings