# Bcs

Fikk denne mailen av Jaakko, som forklarer hva jeg skal kode.  

Please take a look at the attached program that was written with archaic Fortran77 by me. It calculates partial distribution functions by taking into account (cubic) periodic boundary conditions. As a start, make you own code (python?) for the same purpose and calculate PDFs for the structures that David sent. You will need to modify the number of atoms, their labels, number of frames (= 1), and the box size accordingly.


The default bin size is probably too narrow (lots of noise) and you can make it thicker. Note that PDFs are average properties and will look very similar for the different snapshots.


As an extension, you can next filter out the atomic indices for chemical bonds that are shortest / longest. The first peak of PDF corresponds to chemical bonds. You can then take a closer look at those atoms with VMD.


We can make similar srcripts for calculating bond angles, dihedral angles, coordination numbers, etc.
