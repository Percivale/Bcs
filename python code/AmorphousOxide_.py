# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 14:40:34 2022

@author: kajah
"""
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt

class AmorphousOxide:
    def __init__(self, nr_atoms, nr_si, nr_o, xyz_path):
        self.nrAtoms = nr_atoms
        self.nrSi = nr_si
        self.nrO = nr_o
        #self.boxX = boxsize[0]
        #self.boxY = boxsize[1]
        #self.boxZ = boxsize[2]
        self.Si =0
        self.O  = 1
        
        
        xyz_df = pd.read_table(xyz_path, skiprows=1, delim_whitespace=True, nrows = self.nrAtoms+1, names=['atom', 'x', 'y', 'z'])
        boxsize = xyz_df.iloc[0]
        self.boxX = float(boxsize["atom"])
        self.boxY = float(boxsize["x"])
        self.boxZ = float(boxsize["y"])
        #Retrieving data from xyz file and putting them in arrays:
        self.nameArray = list(xyz_df["atom"].to_numpy(dtype = str))[1:]
        
        x = xyz_df["x"].to_numpy()[1:]
        y = xyz_df["y"].to_numpy()[1:]
        z = xyz_df["z"].to_numpy()[1:]
        
        #Implementing periodic boundaries:
        self.xArray = x - self.boxX*np.rint(x/self.boxX)
        self.yArray = y - self.boxY*np.rint(y/self.boxY)
        self.zArray = z - self.boxZ*np.rint(z/self.boxZ)
            
    
    def calc_pdf(self, nhdim, bz):
        adhoc = np.array([self.boxX*self.boxY*self.boxZ/(self.nrSi*(self.nrSi-1))/bz, self.boxX*self.boxY*self.boxZ/(self.nrSi*self.nrO - (self.nrSi + self.nrO))/bz,
                          self.boxX*self.boxY*self.boxZ/(self.nrO*(self.nrO-1))/bz])
        
        count = np.zeros((nhdim, 3))
        dist = np.zeros((nhdim, 3))
        
        for i in range(len(self.xArray)):
            for j in range(len(self.xArray)):
                dx = self.xArray[i] - self.xArray[j] - self.boxX*np.rint((self.xArray[i]-self.xArray[j])/self.boxX)
                dy = self.yArray[i] - self.yArray[j] - self.boxY*np.rint((self.yArray[i]-self.yArray[j])/self.boxY)
                dz = self.zArray[i] - self.zArray[j] - self.boxZ*np.rint((self.zArray[i]-self.zArray[j])/self.boxZ) 
                
                dr = np.sqrt(dx**2 + dy**2 + dz**2)
                
                if dr <= self.boxX/2 and self.nameArray[i] + " " + self.nameArray[j] == "Si Si" and i != j:
                    ll = int(np.rint(dr/bz))
                    count[ll, 0] += 1
                    dist[ll, 0] += dr
                elif dr <= self.boxX/2 and self.nameArray[i] + " " + self.nameArray[j] == "Si O":
                    ll = int(np.rint(dr/bz))
                    count[ll, 1] += 1
                    dist[ll, 1] += dr
                elif dr <= self.boxX/2 and self.nameArray[i] + " " + self.nameArray[j] == "O O" and i != j:
                    ll = int(np.rint(dr/bz))
                    count[ll, 2] += 1
                    dist[ll, 2] += dr
        #---------------------------------------------------------------------
        rbz = np.zeros((len(count), 3))
        i = np.arange(1, len(count)+1, 1)
        
        rbz[:,0] = bz*i
        rbz[:,1] = bz*i
        rbz[:,2] = bz*i
        
        pdf = adhoc*count/(4*np.pi*rbz**2)
        dist = dist/count
        dist = np.nan_to_num(dist)
        
        sisi_pdf = pdf[:,0]
        sio_pdf = pdf[:,1]
        oo_pdf = pdf[:,2]
        
        sisi_dist = dist[:,0]
        sio_dist = dist[:,1]
        oo_dist = dist[:,2]
        
        return sisi_pdf, sio_pdf, oo_pdf, sisi_dist, sio_dist, oo_dist


    
def generate_data(xyz_directory, excel_path): #not done
    frames = 0
    """
    tot_sisi_pdf = []
    tot_sio_pdf = []
    tot_oo_pdf = []
    tot_sisi_dist = []
    tot_sio_dist = []
    tot_oo_dist = []
    occurences_sisi = np.zeros(9600)
    occurences_sio = np.zeros(9600)
    occurences_oo = np.zeros(9600)
    """
    statistics = []
    siosi_angles = np.array([])
    siosi_bonds2 = np.array([])
    osio_angles = np.array([])
    osio_bonds2 = np.array([])
    dihedral_angles = np.array([])
    dihedral_idx = np.array([])
    
    siosi_dr = np.array([])
    osio_dr = np.array([])
    boxid = np.array([])
    
    
    for subdir, dirs, files in os.walk(xyz_directory):
        for file in files:
            filepath = subdir + os.sep + file
            print("Accessing file.... ", file)
            etrap = AmorphousOxide(216, 72, 144, filepath)
            
            
            ##########################################################################
            #Calculating pdfs
            """
            sisi_pdf, sio_pdf, oo_pdf, sisi_dist, sio_dist, oo_dist = etrap.calc_pdf(9600, 0.02)
            
            tot_sisi_pdf.append(sisi_pdf)
            tot_sio_pdf.append(sio_pdf)
            tot_oo_pdf.append(oo_pdf)
            tot_sisi_dist.append(sisi_dist)
            tot_sio_dist.append(sio_dist)
            tot_oo_dist.append(oo_dist)
            
            occurences_sisi[np.where(sisi_dist != 0)] += 1
            occurences_sio[np.where(sio_dist != 0)] += 1
            occurences_oo[np.where(oo_dist != 0)] += 1
            """

            ##########################################################################
            #Bonds
            
            
            cutoffs = [3.44, 2.0, 3.0]
            index, bond_lengths, dr = make_bonds(etrap.xArray, etrap.yArray, etrap.zArray, etrap.nameArray, 
                                                    etrap.boxX, etrap.boxY, etrap.boxZ, cutoffs)
            
            ##########################################################################
            #Bond angles and dihedral angles
            sisisi_idx, siosi_idx, osio_idx = match_bonds(index)
            
            siosi_angle = calc_angle(siosi_idx, dr)
            osio_angle = calc_angle(osio_idx, dr)
            
            dihedrals = get_dihedrals_fast(siosi_idx, index)
            dihed_angles = calc_dihedral(np.array(dihedrals), dr)
            
            r = np.sqrt(np.sum(dr**2, axis = 0))
            
            if frames == 0:  
                siosi_angles = siosi_angle
                siosi_bonds2 = np.array(np.where(siosi_idx != 0)).T
                
                siosi_dr = np.array([r[siosi_bonds2[:,0], siosi_bonds2[:,1]], r[siosi_bonds2[:,1], siosi_bonds2[:,2]]]).T

                
                osio_angles = osio_angle
                osio_bonds2 = np.array(np.where(osio_idx != 0)).T
                osio_dr = np.array([r[osio_bonds2[:,0], osio_bonds2[:,1]], r[osio_bonds2[:,1], osio_bonds2[:,2]]]).T
                
                dihedral_angles = dihed_angles
                dihedral_idx = dihedrals
                
                boxid_siosi = np.full(len(siosi_angle), frames)
                boxid_osio = np.full(len(osio_angle), frames)
                
            else:
                
                temp_siosi = np.array(np.where(siosi_idx != 0)).T
                temp_osio = np.array(np.where(osio_idx != 0)).T
                
                siosi_angles = np.append(siosi_angles, siosi_angle)
                siosi_bonds2 = np.append(siosi_bonds2, temp_siosi, axis = 0)
                siosi_dr = np.append(siosi_dr, np.array([r[temp_siosi[:,0], temp_siosi[:,1]], r[temp_siosi[:,1], temp_siosi[:,2]]]).T, axis = 0)
                
                osio_angles = np.append(osio_angles, osio_angle)
                osio_bonds2 = np.append(osio_bonds2, temp_osio, axis = 0)
                osio_dr = np.append(osio_dr,  np.array([r[temp_osio[:,0], temp_osio[:,1]], r[temp_osio[:,1], temp_osio[:,2]]]).T, axis = 0)
                

                dihedral_angles = np.append(dihedral_angles, dihed_angles)
                dihedral_idx = np.append(dihedral_idx, dihedrals)
                
                boxid_siosi = np.append(boxid_siosi, np.full(len(siosi_angle), frames))
                boxid_osio = np.append(boxid_osio, np.full(len(osio_angle), frames))
                #print(siosi_angles.shape, siosi_dr.shape, siosi_bonds2.shape, osio_angles.shape, osio_dr.shape, osio_bonds2.shape)
    
  
            statistics.append({"name" : file, "bonds" : len(np.array(np.where(index != 0)).T), "angles" : len(siosi_angle) + len(osio_angle), "dihedral angles": len(dihed_angles)})
            
            
            frames += 1
            
    
    angles = np.append(siosi_angles, osio_angles)
    bonds2 = np.append(siosi_bonds2, osio_bonds2, axis = 0)
    dr_all = np.append(siosi_dr, osio_dr, axis = 0)
    name1 = np.append(np.full(len(siosi_angles), "Si"), np.full(len(osio_angles), "O"))
    name2 = np.append(np.full(len(siosi_angles), "O"), np.full(len(osio_angles), "O"))
    boxid = np.append(boxid_siosi, boxid_osio)
    
    print(angles.shape, bonds2.shape, dr_all.shape, boxid.shape)
    
    
    simple_dataset = pd.DataFrame(np.array([boxid, bonds2[:,0], name1, bonds2[:,1], name2, bonds2[:,2], name1, dr_all[:,0], dr_all[:,1], angles]).T, columns = ["Box id", "Atom 1", "Name 1", "Atom 2", "Name 2", "Atom 3", "Name 3", "Bond 1-2", "Bond 2-3", "Angle"])
    simple_dataset["Box id"] = simple_dataset["Box id"].astype(int)
    simple_dataset["Atom 1"] = simple_dataset["Atom 1"].astype(int)
    simple_dataset["Atom 2"] = simple_dataset["Atom 2"].astype(int)
    simple_dataset["Atom 3"] = simple_dataset["Atom 3"].astype(int)
    simple_dataset["Bond 1-2"] = simple_dataset["Bond 1-2"].astype(float)
    simple_dataset["Bond 2-3"] = simple_dataset["Bond 2-3"].astype(float)
    simple_dataset["Angle"] = simple_dataset["Angle"].astype(float)
    
    simple_dataset.to_excel("simple_dataset.xlsx")

        
            
            
    
    
    """
    tot_sisi_pdf = np.sum(np.array(tot_sisi_pdf), axis = 0)#/frames
    tot_sio_pdf = np.sum(np.array(tot_sio_pdf), axis = 0)#/frames
    tot_oo_pdf = np.sum(np.array(tot_oo_pdf), axis = 0)#/frames
    tot_sisi_dist = np.sum(np.array(tot_sisi_dist), axis = 0)#/frames
    tot_sio_dist = np.sum(np.array(tot_sio_dist), axis = 0)#/frames
    tot_oo_dist = np.sum(np.array(tot_oo_dist), axis = 0)#/frames
    
    tot_sisi_pdf = tot_sisi_pdf[tot_sisi_dist!=0]/occurences_sisi[tot_sisi_dist!= 0]
    tot_sisi_dist = tot_sisi_dist[tot_sisi_dist!=0]/occurences_sisi[tot_sisi_dist!= 0]
    tot_sio_pdf = tot_sio_pdf[tot_sio_dist!=0]/occurences_sio[tot_sio_dist!= 0]
    tot_sio_dist = tot_sio_dist[tot_sio_dist!=0]/occurences_sio[tot_sio_dist!= 0]
    tot_oo_pdf = tot_oo_pdf[tot_oo_dist!=0]/occurences_oo[tot_oo_dist!= 0]
    tot_oo_dist = tot_oo_dist[tot_oo_dist!=0]/occurences_oo[tot_oo_dist!= 0]
    """

    

    
    """
    plt.figure(figsize = (2, 5))
    plt.plot(tot_sio_dist, tot_sio_pdf)
    plt.xlim(1.55,1.68)
    plt.ylim(10,75)
    plt.xticks(size = 18)
    plt.yticks(size = 18)
    plt.show()
    
    plt.figure()
    plt.title("Partial Distribution Function", size = 20, y = 1.05)
    plt.plot(tot_sio_dist, tot_sio_pdf, label = "Si-O bonds")
    plt.plot(tot_sisi_dist, tot_sisi_pdf, label = "Si-Si bonds")
    plt.plot(tot_oo_dist, tot_oo_pdf, label = "O-O bonds")
    plt.ylim(0,10)
    plt.xlim(1.5,6)
    plt.xticks(size = 18)
    plt.yticks(size = 18)
    plt.ylabel("Amplitude", size = 18)
    plt.xlabel("Distance [Å]", size = 18)
    plt.legend(fontsize = 16)
    plt.show()
    
    print(dihedral_angles)
    plt.figure()
    plt.title("Dihedral angles", size = 20)
    plt.hist(dihedral_angles)
    plt.xlabel("Angle (Degrees)", size = 18)
    plt.xticks(size = 18)
    plt.yticks(size = 18)
    plt.show()
    """
            
    return statistics, simple_dataset#, dihedral_idx, dihedral_angles

        
def calc_angle(index_matrix, dr):
    
    indexes = np.array(np.where(index_matrix!=0)).T

    index1 = indexes[:, :2]
    index2 = indexes[:, 1:]
    a = np.sqrt(np.sum(dr[:, index1[:,0], index1[:,1]]**2, axis = 0))
    b = np.sqrt(np.sum(dr[:, index2[:,0], index2[:, 1]]**2, axis = 0))
    c = np.sqrt(np.sum(dr[:, index1[:,0], index2[:,1]]**2, axis = 0))
    
    angle = np.rad2deg(np.arccos((a**2 + b**2 - c**2)/(2*a*b))) # Cosine law ##
    return angle    

    
def make_bonds(x_array, y_array, z_array, name_array, boxx, boxy, boxz, cutoffs):
    cut_sisi = cutoffs[0]
    cut_sio = cutoffs[1]
    cut_oo = cutoffs[2]
    
    name_array=np.array(name_array)

    bondlengths = []
    #x_12 = []
    #y_12 = []
    #z_12 = []
    INDEX = np.arange(0,len(name_array),1,dtype=int)
    dist = np.zeros((3, len(name_array), len(name_array)))
    index_2 = np.zeros((len(name_array), len(name_array)), dtype=int)
    
    OO    = np.full(len(name_array),1)
    SIO   = np.full(len(name_array),2)
    SISI  = np.full(len(name_array),3)  

    #chain_index=[]
    bondlengths=[]
    #bondtype=[]


    for i in range(len(name_array)):
        x1 = x_array[i]
        y1 = y_array[i]
        z1 = z_array[i]
        
        x2 = x_array
        y2 = y_array
        z2 = z_array
        
        #find distance between the atoms (with periodic boundary):
        dx = x1 - x2 - boxx*np.rint((x1-x2)/boxx)
        dy = y1 - y2 - boxy*np.rint((y1-y2)/boxy)
        dz = z1 - z2 - boxz*np.rint((z1-z2)/boxz) 
        dr = np.sqrt(dx**2 + dy**2 + dz**2)
        dr[i] = np.inf


        
        if name_array[i] == "Si":
            values_eq=np.logical_and( name_array==name_array[i],dr<cut_sisi)
            values_neq=np.logical_and( name_array!=name_array[i],dr<cut_sio)
            btype = SISI*values_eq + SIO*values_neq
            btype = btype[btype!=0]
            
        elif name_array[i] == "O":
            values_eq = np.logical_and(name_array==name_array[i],dr<cut_oo)
            values_neq = np.full(values_eq.shape, False)
            
            btype = OO[values_eq]
        else:
            print('Somthing is wrong with data file')
            

        val=np.logical_or(values_eq,values_neq)
        index_2[i,INDEX[val]] = btype
        #chain_index.append(INDEX[val])
        bondlengths.append(dr[val])
        #bondtype.append(btype)
        dist[:,i] = np.array([dx, dy, dz])
        
    return index_2, np.array(bondlengths), dist



def match_bonds(index):
    sisisi_index = np.zeros((len(index), len(index), len(index)))
    siosi_index = np.zeros((len(index), len(index), len(index)))
    osio_index = np.zeros((len(index), len(index), len(index)))

    SISISI = 7 #Binary 111
    SIOSI = 5 #binary 101
    OSIO = 2 #binary (0)10
    
    for i in range(len(index)):
        #types = names[i] #all connected atoms bondtype
        indexes = index[i] #all connected atoms index
        

        sisi_idx = np.where(indexes == 3)[0] #Finds all Si atoms connected to first Si atom in chain
        sio_idx = np.where(indexes == 2)[0] #Finds all O atoms connected to first Si atom in chain
            
        for j in range(len(sisi_idx)):
            last_si = np.where(index[sisi_idx[j]] == 3)[0] #finds all Si atoms connected to the middle atom in chain
            last_si = last_si[last_si != i]
            
            sisisi_index[i][sisi_idx[j]][last_si] = SISISI 
            
            
        for j in range(len(sio_idx)):
            last_si = np.where(index.T[sio_idx[j]] == 2)[0] #Finds all Si atoms connected to the middle atom (O-Si bonds are Si-O bonds reversed -> .T)
            last_si = last_si[last_si != i]
            
            siosi_index[i][sio_idx[j]][last_si] = SIOSI 
            
        if len(sio_idx):
            last_o = np.where(indexes == 2)[0].T
            last_o = last_o[last_o != sio_idx.T[0]]
            
            osio_index[sio_idx.T[0], i, last_o] = OSIO
 
    return sisisi_index, siosi_index, osio_index
            

def calc_dihedral(dihed_idx, dr):
    dr12 = dr[:,dihed_idx[:,0], dihed_idx[:, 1]].T
    dr23 = dr[:, dihed_idx[:,1], dihed_idx[:,2]].T
    dr34 = dr[:, dihed_idx[:, 2], dihed_idx[:,3]].T

    x = np.sum(np.linalg.norm(dr23, axis = 1)[:, None]*dr12*np.cross(dr23, dr34), axis = 1)
    y = np.sum(np.cross(dr12, dr23)*np.cross(dr23, dr34), axis = 1)
    
    angle = np.arctan2(x, y)
    return -np.rad2deg(angle)

def get_dihedrals_fast(siosi_idx, idx):
    dihed_idx = []
    SIOSI = 5 #binary 101
    sio_idx = np.array(np.where(idx==2)).T
    chains = np.array(np.where(siosi_idx == SIOSI)).T

    for i in range(len(chains)):
        last_in_chain = chains[i][-1]
        connected_to_last = sio_idx[:,1][sio_idx[:,0] == last_in_chain]
        
        for atom in connected_to_last:
            if atom not in chains[i]:
                dihed_idx.append([chains[i][0], chains[i][1], chains[i][2], atom])
        
    return dihed_idx

def three_ring(sisisi_idx, sisi_idx):
    rings = set([])
    chains = np.array(np.where(sisisi_idx != 0)).T
  
    for i in range(len(chains)):
        chain = chains[i]
        if sisi_idx[chain[0], chain[-1]]:
            rings.add(frozenset(chain))
    
    return rings

def four_ring(three_ring, sisisi_idx, sisi_idx):
    rings = set([])
    chains = np.array(np.where(sisisi_idx != 0)).T
    
    for i in range(len(chains)):
        chain = chains[i].flatten()
        
        last_atom = np.array(np.where(sisisi_idx[chain[-1], :, chain[0]] != 0)).flatten()
        #print(chain, last_atom)
        if frozenset(chain) not in three_ring:
            for j in range(len(last_atom)):
                if last_atom[j] not in chain:
                    #print(chain, last_atom[j])
                    rings.add(frozenset([chain[0], chain[1], chain[2], last_atom[j]]))
    return rings

def five_ring(three_ring, four_ring, sisisi_idx, sisi_idx):
    rings = set([])
    chains = np.array(np.where(sisisi_idx != 0)).T
    
    for i in range(len(chains)):
        chain = chains[i].flatten()
        connections_right = np.array(np.where(sisisi_idx[chain[0], :, :] != 0)).T
        connections_left = np.array(np.where(sisisi_idx[chain[1], chain[-1], :] != 0)).flatten()

        if len(connections_left)>0 and len(connections_right) > 0:
            
            if frozenset(chain) not in three_ring:

                for j in range(len(connections_right)):
                    for k in range(len(connections_left)):
    
                        if (sisi_idx[connections_left[k], connections_right[j, 0]] != 0 and 
                            connections_left[k] not in chain and 
                            connections_right[j,0] not in chain and
                            sisi_idx[connections_right[j,0], chain[1]] == 0 and
                            sisi_idx[chain[1], connections_left[k]] == 0 and
                            sisi_idx[chain[0], connections_left[k]] == 0 and
                            sisi_idx[connections_right[j,0], chain[2]] == 0):
                            if (frozenset([connections_left[k], chain[0], chain[1]]) not in three_ring or
                                frozenset([chain[0], chain[1], connections_right[j,0]]) not in three_ring or
                                frozenset([connections_left[k], chain[0], connections_right[j,0]]) not in three_ring or
                                frozenset([connections_right[j,0], chain[0], chain[1]]) not in three_ring):
                                if (frozenset([connections_right[j,0], chain[0], chain[1], chain[2]]) not in four_ring or
                                    frozenset([chain[0], chain[1], chain[2], connections_left[k]]) not in four_ring or 
                                    frozenset([chain[1], chain[2], connections_left[k], connections_right[j,0]]) not in four_ring):
                                    #print(chain, connections_right[j])
                                    if 215 in [connections_right[j, 0],chain[0], chain[1], chain[2], connections_left[k]]:
                                        print([connections_right[j, 0],chain[0], chain[1], chain[2], connections_left[k]])
                                    rings.add(frozenset([connections_right[j, 0],chain[0], chain[1], chain[2], connections_left[k]]))
    return rings

def six_ring(rings3, rings4, rings5, sisisi_idx):
    rings = set([])
    chains = np.array(np.where(sisisi_idx != 0)).T
    
    for i in range(len(chains)):
        chain = chains[i].flatten()
        connected_right = np.array(np.where(sisisi_idx[chain[0], :, :] != 0)).T
        connected_left = np.array(np.where(sisisi_idx[chain[-1], :, :] != 0)).T
        if connected_left.size != 0 and connected_right.size != 0:
            last_left = connected_left[:,1]
            last_right = connected_right[:,1]
            if frozenset(chain) not in rings3:
                for j in range(len(last_left)):
                    connected = np.array(np.where(last_right == last_left[j])).flatten()
                    
                    if connected.size !=0:
                        #print(connected)
                        for k in range(len(connected)):
                            if connected_left[j,0] != connected_right[connected[k],0] and connected_left[j,0] not in chain and connected_right[connected[k],0] not in chain and connected_right[connected[k],1] not in chain:
                                if (frozenset([connected_left[j,0], chain[0], chain[1]]) in rings3 or 
                                    frozenset([chain[0], chain[1], connected_right[connected[k], 0]]) in rings3 or 
                                    frozenset([chain[1], chain[2], connected_right[connected[k],0]]) in rings3 or 
                                    frozenset([chain[2], connected_right[connected[k],0], connected_right[connected[k], 1]]) in rings3 or 
                                    frozenset([connected_right[connected[k],0], connected_right[connected[k],1], connected_left[j,0]]) in rings3 or
                                    frozenset([connected_right[connected[k],1], connected_left[j,0], chain[0]]) in rings3):
                                    continue
                                
                                else:
                                    if (frozenset([connected_left[j,0], chain[0], chain[1], chain[2]]) in rings4 or
                                        frozenset([chain[0], chain[1], chain[2], connected_right[connected[k], 0]]) in rings4 or
                                        frozenset([chain[1], chain[2], connected_right[connected[k],0], connected_right[connected[k],1]]) in rings4 or
                                        frozenset([chain[2], connected_right[connected[k],0], connected_right[connected[k],1], connected_left[j,0]]) in rings4 or
                                        frozenset([connected_right[connected[k],0], connected_right[connected[k],1], connected_left[j,0], chain[0]]) in rings4 or
                                        frozenset([connected_right[connected[k],1], connected_left[j,0], chain[0], chain[1]]) in rings4):
                                        continue
                                    else:
                                        if (frozenset([connected_left[j,0], chain[0], chain[1], chain[2], connected_right[connected[k],0]]) in rings5 or 
                                            frozenset([chain[0], chain[1], chain[2], connected_right[connected[k],0], connected_right[connected[k],1]]) in rings5 or
                                            frozenset([chain[1], chain[2], connected_right[connected[k],0], connected_right[connected[k],1], connected_left[j,0]]) in rings5 or
                                            frozenset([chain[2], connected_right[connected[k],0], connected_right[connected[k],1], connected_left[j,0], chain[0]]) in rings5 or
                                            frozenset([connected_right[connected[k],0], connected_right[connected[k],1], connected_left[j,0], chain[0], chain[1]]) in rings5 or
                                            frozenset([connected_right[connected[k],1], connected_left[j,0], chain[0], chain[1], chain[2]]) in rings5):
                                            continue
                                        else:
                                            rings.add(frozenset([connected_left[j,0], chain[0], chain[1], chain[2], connected_right[connected[k],0], connected_right[connected[k],1]]))
                    
    return rings

def seven_ring(rings3, rings4, rings5, rings6, sisisi_idx):
    rings = set([])
    chains = np.array(np.where(sisisi_idx != 0)).T
    
    for i in range(len(chains)):
        chain = chains[i].flatten()
        chain = chains[i].flatten()
        connected_right = np.array(np.where(sisisi_idx[chain[0], :, :] != 0)).T
        connected_left = np.array(np.where(sisisi_idx[chain[-1], :, :] != 0)).T
        if frozenset(chain) not in rings3:
            if len(connected_left)>0 and len(connected_right)>0:
                for j in range(len(connected_left)):
                    for k in range(len(connected_right)):
                        if sisisi_idx[connected_left[j,0], connected_left[j, 1], connected_right[k,-1]] != 0:
                            
                            if connected_left[j,0] not in chain and connected_left[j, 1] not in chain and connected_right[k,0] not in chain and connected_right[k,1] not in chain:
                                
                                if (frozenset([connected_left[j,0], connected_left[j, 1], chain[0]]) in rings3 or
                                    frozenset([connected_left[j, 1], chain[0], chain[1]]) in rings3 or
                                    frozenset([chain[0], chain[1], connected_right[k, 0]]) in rings3 or
                                    frozenset([chain[1], connected_right[k, 0], connected_right[k, 1]]) in rings3 or
                                    frozenset([connected_right[k, 0], connected_right[k, 1], connected_left[j,0]]) in rings3 or
                                    frozenset([connected_right[k, 1], connected_left[j,0], connected_left[j,1]]) in rings3):
                                    continue
                                else:
                                    #print(connected_left[j], chain, connected_right[k])
                                    if(frozenset([connected_left[j, 0], connected_left[j, 1], chain[0], chain[1]]) in rings4 or
                                       frozenset([connected_left[j, 1], chain[0], chain[1], chain[2]]) in rings4 or
                                       frozenset([chain[0], chain[1], chain[2], connected_right[k, 0]]) in rings4 or
                                       frozenset([chain[1], chain[2], connected_right[k, 0], connected_right[k,1]]) in rings4 or
                                       frozenset([chain[2], connected_right[k, 0], connected_right[k,1], connected_left[j, 0]]) in rings4 or
                                       frozenset([connected_right[k, 0], connected_right[k,1], connected_left[j, 0], connected_left[j, 1]]) in rings4 or
                                       frozenset([connected_right[k, 1], connected_left[j, 0], connected_left[j, 1], chain[0]]) in rings4):
                                       continue
                                    else:
                                        if(frozenset([connected_left[j, 0], connected_left[j, 1], chain[0], chain[1], chain[2]]) in rings5 or
                                           frozenset([connected_left[j, 1], chain[0], chain[1], chain[2], connected_right[k, 0]]) in rings5 or
                                           frozenset([chain[0], chain[1], chain[2], connected_right[k, 0], connected_right[k, 1]]) in rings5 or
                                           frozenset([chain[1], chain[2], connected_right[k, 0], connected_right[k, 1], connected_left[j, 0]]) in rings5 or
                                           frozenset([chain[2], connected_right[k, 0], connected_right[k, 1], connected_left[j, 0], connected_left[j, 1]]) in rings5 or
                                           frozenset([connected_right[k, 0], connected_right[k, 1], connected_left[j, 0], connected_left[j, 1], chain[0]]) in rings5 or
                                           frozenset([connected_right[k, 1], connected_left[j, 0], connected_left[j, 1], chain[0], chain[1]]) in rings5):
                                            continue
                                        else:
                                            if (frozenset([connected_left[j, 0], connected_left[j, 1], chain[0], chain[1], chain[2], connected_right[k, 0]]) in rings6 or
                                                frozenset([connected_left[j, 1], chain[0], chain[1], chain[2], connected_right[k, 0], connected_right[k, 1]]) in rings6 or
                                                frozenset([chain[0], chain[1], chain[2], connected_right[k, 0], connected_right[k, 1], connected_left[j,0]]) in rings6 or
                                                frozenset([chain[1], chain[2], connected_right[k, 0], connected_right[k, 1], connected_left[j,0], connected_left[j,1]]) in rings6 or
                                                frozenset([chain[2], connected_right[k, 0], connected_right[k, 1], connected_left[j,0], connected_left[j,1], chain[0]]) in rings6 or
                                                frozenset([connected_right[k, 0], connected_right[k, 1], connected_left[j,0], connected_left[j,1], chain[0], chain[1]]) in rings6 or
                                                frozenset([connected_right[k, 1], connected_left[j,0], connected_left[j,1], chain[0], chain[1], chain[2]]) in rings6):
                                                continue
                                            else:
                                                if i <10:
                                                    print(connected_left[j, 0], connected_left[j, 1], chain[0], chain[1], chain[2], connected_right[k, 0], connected_right[k,1])
                                                rings.add(frozenset([connected_left[j, 0], connected_left[j, 1], chain[0], chain[1], chain[2], connected_right[k, 0], connected_right[k,1]]))        
    return rings

def eight_ring(rings3, rings4, rings5, rings6, rings7, sisisi_idx):
    rings = set([])
    chains = np.array(np.where(sisisi_idx != 0)).T
    
    for i in range(len(chains)):
        chain = chains[i].flatten()

        connected_right = np.array(np.where(sisisi_idx[chain[0], :, :] != 0)).T
        connected_left = np.array(np.where(sisisi_idx[chain[-1], :, :] != 0)).T
        
        if frozenset(chain) in rings3 or connected_right.size == 0 or connected_left.size == 0:
            continue
        else:
        
            for j in range(len(connected_left)):               
                for k in range(len(connected_right)):
                    
                    if connected_right[k,0] not in chain and connected_right[k, 1] not in chain and connected_left[j,0] not in chain and connected_left[j,1] not in chain:
                        last_chain = np.array(np.where(sisisi_idx[connected_right[k,-1], :, connected_left[j,-1]])).flatten()
                        if last_chain.size != 0:
                            for l in range(len(last_chain)):
                                if last_chain[l] not in chain and last_chain[l] not in connected_left[j] and last_chain[l] not in connected_right[k]:
                                    if (frozenset([connected_left[j,0], connected_left[j, 1], chain[0]]) in rings3 or
                                        frozenset([connected_left[j, 1], chain[0], chain[1]]) in rings3 or
                                        frozenset([chain[0], chain[1], connected_right[k, 0]]) in rings3 or
                                        frozenset([chain[1], connected_right[k, 0], connected_right[k, 1]]) in rings3 or
                                        frozenset([connected_right[k, 0], connected_right[k, 1], last_chain[l]]) in rings3 or
                                        frozenset([connected_right[k, 1], last_chain[l], connected_left[j,0]]) in rings3 or
                                        frozenset([last_chain[l], connected_left[j,0], connected_left[j,1]]) in rings3 or
                                        frozenset([chain[0], connected_right[k,0], connected_right[k,1]]) in rings3 or
                                        frozenset([connected_right[k, 1], connected_left[j,0], connected_left[j,1]]) in rings3):
                                        #print(3)
                                        continue
                                    else:
                                        if(frozenset([connected_left[j, 0], connected_left[j, 1], chain[0], chain[1]]) in rings4 or
                                           frozenset([connected_left[j, 1], chain[0], chain[1], chain[2]]) in rings4 or
                                           frozenset([chain[0], chain[1], chain[2], connected_right[k, 0]]) in rings4 or
                                           frozenset([chain[1], chain[2], connected_right[k, 0], connected_right[k,1]]) in rings4 or
                                           frozenset([chain[2], connected_right[k, 0], connected_right[k,1], last_chain[l]]) in rings4 or
                                           frozenset([chain[2], connected_right[k, 0], connected_right[k,1], connected_left[j, 0]]) in rings4 or
                                           frozenset([connected_right[k, 0], connected_right[k,1], last_chain[l], connected_left[j, 0]]) in rings4 or
                                           frozenset([connected_right[k, 0], connected_right[k,1], connected_left[j, 0], connected_left[j, 1]]) in rings4 or
                                           frozenset([connected_right[k,1], last_chain[l], connected_left[j, 0], connected_left[j,1]]) in rings4 or
                                           frozenset([last_chain[l], connected_left[j, 0], connected_left[j,1], chain[0]]) in rings4 or
                                           frozenset([connected_right[k, 1], connected_left[j, 0], connected_left[j, 1], chain[0]]) in rings4 or
                                           frozenset([connected_left[j,0], connected_left[j,1], chain[1], chain[2]]) in rings4):
                                           #print(4)
                                           continue
                                        else:
                                            if(frozenset([connected_left[j, 0], connected_left[j, 1], chain[0], chain[1], chain[2]]) in rings5 or
                                               frozenset([connected_left[j, 1], chain[0], chain[1], chain[2], connected_right[k, 0]]) in rings5 or
                                               frozenset([chain[0], chain[1], chain[2], connected_right[k, 0], connected_right[k, 1]]) in rings5 or
                                               frozenset([chain[1], chain[2], connected_right[k, 0], connected_right[k, 1], last_chain[l]]) in rings5 or
                                               frozenset([chain[2], connected_right[k, 0], connected_right[k, 1], last_chain[l], connected_left[j, 0]]) in rings5 or
                                               frozenset([connected_right[k, 0], connected_right[k, 1], last_chain[l], connected_left[j, 0], connected_left[j,1]]) in rings5 or
                                               frozenset([connected_right[k, 1], last_chain[l], connected_left[j, 0], connected_left[j,1], chain[0]]) in rings5 or
                                               frozenset([last_chain[l], connected_left[j, 0], connected_left[j,1], chain[0], chain[1]]) in rings5 or
                                               frozenset([chain[1], chain[2], connected_right[k, 0], connected_right[k, 1], connected_left[j, 0]]) in rings5 or
                                               frozenset([chain[2], connected_right[k, 0], connected_right[k, 1], connected_left[j, 0], connected_left[j, 1]]) in rings5 or
                                               frozenset([connected_right[k, 0], connected_right[k, 1], connected_left[j, 0], connected_left[j, 1], chain[0]]) in rings5 or
                                               frozenset([connected_right[k, 1], connected_left[j, 0], connected_left[j, 1], chain[0], chain[1]]) in rings5 or
                                               frozenset([connected_right[k, 0], connected_right[k,1], chain[0], chain[1], chain[2]]) in rings5):
                                               #print(5)29 38 10 13 0
                                               continue
                                           
                                            else:
                                                if (frozenset([connected_left[j, 0], connected_left[j, 1], chain[0], chain[1], chain[2], connected_right[k, 0]]) in rings6 or
                                                    frozenset([connected_left[j, 1], chain[0], chain[1], chain[2], connected_right[k, 0], connected_right[k, 1]]) in rings6 or
                                                    frozenset([chain[0], chain[1], chain[2], connected_right[k, 0], connected_right[k, 1], connected_left[j,0]]) in rings6 or
                                                    frozenset([chain[0], chain[1], chain[2], connected_right[k, 0], connected_right[k, 1], last_chain[l]]) in rings6 or
                                                    frozenset([chain[1], chain[2], connected_right[k, 0], connected_right[k, 1], last_chain[l], connected_left[j,0]]) in rings6 or
                                                    frozenset([chain[2], connected_right[k, 0], connected_right[k, 1], last_chain[l], connected_left[j,0], connected_left[j,1]]) in rings6 or
                                                    frozenset([connected_right[k, 0], connected_right[k, 1], last_chain[l], connected_left[j,0], connected_left[j,1], chain[0]]) in rings6 or
                                                    frozenset([connected_right[k, 1], last_chain[l], connected_left[j,0], connected_left[j,1], chain[0], chain[1]]) in rings6 or
                                                    frozenset([last_chain[l], connected_left[j,0], connected_left[j,1], chain[0], chain[1], chain[2]]) in rings6 or
                                                    frozenset([chain[1], chain[2], connected_right[k, 0], connected_right[k, 1], connected_left[j,0], connected_left[j,1]]) in rings6 or
                                                    frozenset([chain[2], connected_right[k, 0], connected_right[k, 1], connected_left[j,0], connected_left[j,1], chain[0]]) in rings6 or
                                                    frozenset([connected_right[k, 0], connected_right[k, 1], connected_left[j,0], connected_left[j,1], chain[0], chain[1]]) in rings6 or
                                                    frozenset([connected_right[k, 1], connected_left[j,0], connected_left[j,1], chain[0], chain[1], chain[2]]) in rings6):
                                                    #print(6)
                                                    continue
                                                else:
                                                    if (frozenset([connected_left[j, 0], connected_left[j, 1], chain[0], chain[1], chain[2], connected_right[k, 0], connected_right[k,1]]) in rings6 or
                                                        frozenset([connected_left[j, 1], chain[0], chain[1], chain[2], connected_right[k, 0], connected_right[k, 1], last_chain[l]]) in rings6 or
                                                        frozenset([chain[0], chain[1], chain[2], connected_right[k, 0], connected_right[k, 1], last_chain[l], connected_left[j,0]]) in rings6 or
                                                        frozenset([chain[1], chain[2], connected_right[k, 0], connected_right[k, 1], last_chain[l], connected_left[j,0], connected_left[j, 1]]) in rings6 or
                                                        frozenset([chain[2], connected_right[k, 0], connected_right[k, 1], last_chain[l], connected_left[j,0], connected_left[j,1], chain[0]]) in rings6 or
                                                        frozenset([connected_right[k, 0], connected_right[k, 1], last_chain[l], connected_left[j,0], connected_left[j,1], chain[0], chain[1]]) in rings6 or
                                                        frozenset([connected_right[k, 1], last_chain[l], connected_left[j,0], connected_left[j,1], chain[0], chain[1], chain[1]]) in rings6 or
                                                        frozenset([last_chain[l], connected_left[j,0], connected_left[j,1], chain[0], chain[1], chain[2], connected_right[k,0]]) in rings6 or
                                                        frozenset([chain[1], chain[2], connected_right[k, 0], connected_right[k, 1], connected_left[j,0], connected_left[j,1], last_chain[l]]) in rings6 or
                                                        frozenset([chain[2], connected_right[k, 0], connected_right[k, 1], connected_left[j,0], connected_left[j,1], last_chain[l], chain[0]]) in rings6 or
                                                        frozenset([connected_right[k, 0], connected_right[k, 1], connected_left[j,0], connected_left[j,1], last_chain[l], chain[0], chain[1]]) in rings6 or
                                                        frozenset([connected_right[k, 1], last_chain[l], connected_left[j,0], connected_left[j,1], chain[0], chain[1], chain[2]]) in rings6):
                                                        continue
                                                    else:
                                                        
                                                        rings.add(frozenset([connected_left[j,0], connected_left[j,1], chain[0], chain[1], chain[2], connected_right[k, 0], connected_right[k, 1], last_chain[l]]))
    return rings

def nine_ring(rings3, rings4, rings5, rings6, rings7, rings8, sisisi_idx):
    rings = set([])
    chains = np.array(np.where(sisisi_idx != 0)).T
    for i in range(len(chains)):
        chain = chains[i].flatten()
        
        connected_right = np.array(np.where(sisisi_idx[:, :, :] != 0)).T
        connected_left = np.array(np.where(sisisi_idx[:, :, :] != 0)).T
        
        for j in range(len(connected_right)):
            
            if sisisi_idx[chain[2], connected_right[j,0], connected_right[j, 1]] != 0 and connected_right[j, 0] not in chain and connected_right[j, 1] not in chain and connected_right[j, 2] not in chain:
                
                for k in range(len(connected_left)):                    
                    if (sisisi_idx[chain[0], connected_left[k,2], connected_left[k,1]] != 0 and 
                        sisisi_idx[connected_right[j,2], connected_left[k,0], connected_left[k,1]] and
                        connected_left[k, 0] not in chain and 
                        connected_left[k, 1] not in chain and 
                        connected_left[k, 2] not in chain and
                        
                        connected_left[k, 0] != connected_right[j, 0] and 
                        connected_left[k, 1] != connected_right[j, 0] and 
                        connected_left[k, 1] != connected_right[j, 1] and
                        
                        connected_left[k, 0] != connected_right[j, 2] and 
                        connected_left[k, 1] != connected_right[j, 2] and           
                        connected_left[k, 2] != connected_right[j, 2] and
                        
                        connected_left[k, 2] != connected_right[j, 0] and 
                        connected_left[k, 2] != connected_right[j, 0] and 
                        connected_left[k, 2] != connected_right[j, 1]):

                        if sisisi_idx[connected_right[j, 2], connected_left[k, 0], connected_left[k,1]] != 0:

                            if (frozenset([connected_left[j,0], connected_left[j, 1], chain[0]]) in rings3 or
                                frozenset(connected_left[j]) in rings3 or
                                frozenset(connected_right[k]) in rings3 or
                                frozenset([connected_left[j, 1], chain[0], chain[1]]) in rings3 or
                                frozenset([chain[0], chain[1], connected_right[k, 0]]) in rings3 or
                                frozenset([chain[1], connected_right[k, 0], connected_right[k, 1]]) in rings3 or
                                frozenset([chain[0], connected_right[k,0], connected_right[k,1]]) in rings3 or
                                frozenset([connected_right[k, 1], connected_left[j,0], connected_left[j,1]]) in rings3):
                                #print(3)
                                continue
                            else:
                                if(frozenset([connected_left[j, 0], connected_left[j, 1], chain[0], chain[1]]) in rings4 or
                                   frozenset([connected_left[j, 1], chain[0], chain[1], chain[2]]) in rings4 or
                                   frozenset([chain[0], chain[1], chain[2], connected_right[k, 0]]) in rings4 or
                                   frozenset([chain[1], chain[2], connected_right[k, 0], connected_right[k,1]]) in rings4 or
                                   frozenset([chain[2], connected_right[k, 0], connected_right[k,1], connected_left[j, 0]]) in rings4 or
                                   frozenset([connected_right[k, 0], connected_right[k,1], connected_left[j, 0], connected_left[j, 1]]) in rings4 or
                                   frozenset([connected_right[k, 1], connected_left[j, 0], connected_left[j, 1], chain[0]]) in rings4 or
                                   frozenset([connected_left[j,0], connected_left[j,1], chain[1], chain[2]]) in rings4):
                                   #print(4)
                                   continue
                                else:
                                    if(frozenset([connected_left[j, 0], connected_left[j, 1], chain[0], chain[1], chain[2]]) in rings5 or
                                       frozenset([connected_left[j, 1], chain[0], chain[1], chain[2], connected_right[k, 0]]) in rings5 or
                                       frozenset([chain[0], chain[1], chain[2], connected_right[k, 0], connected_right[k, 1]]) in rings5 or
                                       frozenset([chain[1], chain[2], connected_right[k, 0], connected_right[k, 1], connected_left[j, 0]]) in rings5 or
                                       frozenset([chain[2], connected_right[k, 0], connected_right[k, 1], connected_left[j, 0], connected_left[j, 1]]) in rings5 or
                                       frozenset([connected_right[k, 0], connected_right[k, 1], connected_left[j, 0], connected_left[j, 1], chain[0]]) in rings5 or
                                       frozenset([connected_right[k, 1], connected_left[j, 0], connected_left[j, 1], chain[0], chain[1]]) in rings5 or
                                       frozenset([connected_right[k, 0], connected_right[k,1], chain[0], chain[1], chain[2]]) in rings5):
                                       #print(5)29 38 10 13 0
                                       continue
                                   
                                    else:
                                        if (frozenset([connected_left[j, 0], connected_left[j, 1], chain[0], chain[1], chain[2], connected_right[k, 0]]) in rings6 or
                                            frozenset([connected_left[j, 1], chain[0], chain[1], chain[2], connected_right[k, 0], connected_right[k, 1]]) in rings6 or
                                            frozenset([chain[0], chain[1], chain[2], connected_right[k, 0], connected_right[k, 1], connected_left[j,0]]) in rings6 or
                                            frozenset([chain[1], chain[2], connected_right[k, 0], connected_right[k, 1], connected_left[j,0], connected_left[j,1]]) in rings6 or
                                            frozenset([chain[2], connected_right[k, 0], connected_right[k, 1], connected_left[j,0], connected_left[j,1], chain[0]]) in rings6 or
                                            frozenset([connected_right[k, 0], connected_right[k, 1], connected_left[j,0], connected_left[j,1], chain[0], chain[1]]) in rings6 or
                                            frozenset([connected_right[k, 1], connected_left[j,0], connected_left[j,1], chain[0], chain[1], chain[2]]) in rings6):
                                            #print(6)
                                            continue
                                        else:
                                            if (frozenset([connected_left[j, 0], connected_left[j, 1], chain[0], chain[1], chain[2], connected_right[k, 0], connected_right[k,1]]) in rings6):
                                                continue
                                            else:
                                                #if i <5:
                                                    #print(connected_left[j], chain, connected_right[k])
                                                rings.add(frozenset([connected_left[j, 0], connected_left[j, 1], connected_left[j, 2], chain[0], chain[1], chain[2], connected_right[k, 0], connected_right[k,1], connected_right[k,2]]))  
    return rings
    



                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        