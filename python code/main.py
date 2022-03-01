# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 16:04:54 2022

@author: kajah
"""
#%%
import AmorphousOxide_ as ao
import numpy as np
import time
import matplotlib.pyplot as plt

xyz_path = "C:\\Users\\kajah\\ntnu\\bachelor_oppg\\xyz_files\\etrap_015.xyz"

etrap = ao.AmorphousOxide(216, 72, 144, xyz_path)


cutoffs = [3.39, 2.0, 3.0] 
#cut off lengths for bonds, unit Å. Distances between 
#atoms that are shorter or equal to these are chemical bonds
#or next-nearest neighbor (one atom and two chemical bonds in between).
#First element is for Si Si bonds, second for Si O bonds, last for O O bonds. 

sisi_pdf, sio_pdf, oo_pdf, sisi_dist, sio_dist, oo_dist = etrap.calc_pdf(9600, 0.05)

plt.figure()
plt.plot(sisi_dist, sisi_pdf, "-o")
plt.show()

plt.figure()
plt.plot(sio_dist, sio_pdf, "-o")
plt.show()

plt.figure()
plt.plot(oo_dist, oo_pdf, "-o")
plt.show()

#sisisi_pdf, siosi_pdf, osio_pdf, ooo_pdf, sisisi_ang, siosi_ang, osio_ang, ooo_ang = etrap.calc_pdf_angle(9600, 0.05)

#plt.figure()
#plt.plot(sisisi_ang, sisisi_pdf)
#plt.show()

#Finding bonds between atoms
start = time.time()
print("Finding bonds....")
index, bond_lengths, dr = ao.make_bonds(etrap.xArray, etrap.yArray, etrap.zArray, etrap.nameArray, 
                                        etrap.boxX, etrap.boxY, etrap.boxZ, cutoffs)
end = time.time()

cutoffs = [3.39, 2.0, 3.0] 
index1, bond_lengths1, dr1 = ao.make_bonds(etrap.xArray, etrap.yArray, etrap.zArray, etrap.nameArray, 
                                        etrap.boxX, etrap.boxY, etrap.boxZ, cutoffs)

print(np.where(index != index1))
print("Time used to find bonds: ", end-start)
print(len(np.array(np.where(index != 0)).T))
#print(index[0], names[0], bond_lengths[0])
print("Bonds")
#print(pd.DataFrame(bonds[:5], columns = ["Index", "Name", "Bond length"]))
#######################
#%%
#Finding angles between bonds:


start = time.time()
print("\n\nFinding bond angles....")
a1, a2, a3 = ao.match_bonds(index)
siosi_angle = ao.calc_angle(a2, dr)
osio_angle = ao.calc_angle(a3, dr)
print(siosi_angle[:5])
print(osio_angle[:5])

end = time.time()
print("Time used to find bond angles: ", end-start)

#########################


#Finding dihedral angles:
start = time.time()
print("\n\nFinding dihedral angles....")
dihedrals = ao.get_dihedrals_fast(a2, index)
print(len(dihedrals))
end = time.time()
print("Time used to find dihedrals: ", end-start)
dihed_angles = ao.calc_dihedral(np.array(dihedrals), dr)
print(dihed_angles.shape)
print(np.array([dihedrals[:10], dihed_angles[:10]]).T)



#########################
#Finding Si-Si rings up to size 6:


print("\n\nFinding rings of size 3")
start = time.time()
sisi_idx = (index == 3)
rings3 = ao.three_ring(a1, sisi_idx) 
print(len(rings3))
end = time.time()
print("Time used: ", end-start)

print("\n\nFinding rings of size 4")
start = time.time()
rings4 = ao.four_ring(rings3, a1, sisi_idx)
print(len(rings4))
end = time.time()
print("Time used: ", end-start)

print("\n\nFinding rings of size 5")
start = time.time()
rings5 = ao.five_ring(rings3, rings4, a1, sisi_idx) 
print(len(rings5))
end = time.time()
print("Time used: ", end-start)

print("\n\nFinding rings of size 6")
start = time.time()
rings6 = ao.six_ring(rings3, rings4, rings5, a1) 
print(len(rings6))
end = time.time()
print("Time used: ", end-start)

"""
print("\n\nFinding rings of size 7")
start = time.time()
rings7 = ao.seven_ring(rings3, rings4, rings5, rings6, a1) 
print(len(rings7))
end = time.time()
print("Time used: ", end-start)


print("\n\nFinding rings of size 8")
start = time.time()
rings8 = ao.eight_ring(rings3, rings4, rings5, rings6, rings7, a1) 
#print(rings8)
print(len(rings8))
end = time.time()
print("Time used: ", end-start)

print("\n\nFinding rings of size 9")
start = time.time()
rings9 = ao.nine_ring(rings3, rings4, rings5, rings6, rings7, rings8, a1) 
#print(rings8)
print(len(rings9))
end = time.time()
print("Time used: ", end-start)
"""
# %%
