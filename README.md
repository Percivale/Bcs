# Compilere fortran code
Open a WSL (ubuntu) terminal window, move to the directory where your files are located. If you have the files at 'C:\Users\kajahh\git_repo' do
```
Terminal> cd /mnt/c/Users/kajahh/git_repo/Bcs/f_code
Terminal> f77 gofr.f -o gofr
Terminal> ./gofr
```
# C++ projects
C++ projects are most convenient by the use of [Cmake](https://cmake.org/download/). Download and install latest version for Windows, and also for WSL/Ubuntu (don't use sudo apt-get, install from source)

## Eigen
Pull latest version of [Eigen](https://gitlab.com/libeigen/eigen). Read the [INSTALL](https://gitlab.com/libeigen/eigen/-/blob/master/INSTALL) instructions using Cmake.

## run Cmake in this directory
do

```
Terminal> mkdir build
Terminal> cd build
Terminal> cmake ..
Terminal> make
```
The executable code is located in `bin/`



# Bcs

Fikk denne mailen av Jaakko, som forklarer hva jeg skal kode.  

Please take a look at the attached program that was written with archaic Fortran77 by me. It calculates partial distribution functions by taking into account (cubic) periodic boundary conditions. As a start, make you own code (python?) for the same purpose and calculate PDFs for the structures that David sent. You will need to modify the number of atoms, their labels, number of frames (= 1), and the box size accordingly.


The default bin size is probably too narrow (lots of noise) and you can make it thicker. Note that PDFs are average properties and will look very similar for the different snapshots.


As an extension, you can next filter out the atomic indices for chemical bonds that are shortest / longest. The first peak of PDF corresponds to chemical bonds. You can then take a closer look at those atoms with VMD.


We can make similar srcripts for calculating bond angles, dihedral angles, coordination numbers, etc.
