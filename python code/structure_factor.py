# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 11:46:47 2022

@author: kajah
"""
import matplotlib.pyplot as plt
import AmorphousOxide_ as ao
import numpy as np

xyz_path = "C:\\Users\\kajah\\git_repo\\Bcs\\xyz_files_new\\xyz"

sfac, q = ao.analyze_structure_factor(xyz_path)
#%%
exp = np.genfromtxt("C:\\Users\\kajah\\git_repo\\Bcs\\datasets\\exp_strucfac.csv", delimiter = ",")

plt.figure()
plt.title("Structure factor of a-SiO$_2$ structures", size = 22, y = 1.05)
plt.plot(q[17:], sfac[17:], color = "royalblue", linewidth = 4, label = "Modelled structures")
plt.plot(exp[:, 0], exp[:,1],"-o", color = "darkorange", label = "Experimental")
plt.xlabel("Q [Å$^{-1}$]", size = 16)
plt.ylabel("S(Q)", size = 16)
plt.xticks(size = 14)
plt.yticks(size = 14)
plt.legend()
plt.show()

"""
xyz_path = "C:\\Users\\kajah\\git_repo\\Bcs\\xyz_files_new\\xyz\\asio2_001_post.xyz"

test = ao.AmorphousOxide(216, 72, 144, xyz_path)

sfac, q = test.structure_factor_fast(q0 = 1, qn = 15, dq = 0.1)

#%%
#print(sfac)
plt.figure()
plt.title("Structure factor of a-SiO$_2$ model")
plt.plot(q, sfac)
plt.xlabel("Q [Å$^{-1}$]")
plt.ylabel("S(Q)")
plt.show()

"""