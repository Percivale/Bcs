# Machine Learning Identification of Defect Fingerprints in Amorphous Silicon Dioxide

Amorphous materials have a non-crystalline nature, where the only structural requirement is that there is an approximately constant separation of nearest-neighbour atoms \cite{kittel}. Thus, the atoms can be distributed in such a way that they keep an approximate distance between their nearest neighbour, but without the long-range periodicity that is characteristic of crystals. The short-range order of the amorphous structure ensures distributions of properties such as bond lengths and angles, instead of one or two constant values across the entire structure. These distributions can vary from structure to structure depending on the presence of microstructures and interfaces within the material.

Amorphous silicon dioxide consists of SiO$_4$ tetrahedra, where the silicon is in the centre and the oxygen atoms are placed in the corners, see Figure \ref{fig:asio2}(a). Amorphous solids, among them silicon dioxide, play a major role in the development of many modern microelectronics \cite{street1999technology}, \cite{sushko2005structure}. Applications of amorphous solids include a wide range of highly sensitive transducers, sensing devices, and various converters \cite{zolotukhin1990amorphous}, as well as RAM devices and gate insulators in electronic devices \cite{milardovich2021machine}, \cite{idintrinsic}. A common application of amorphous silica is in optical fibres, which are used in a wide range of technologies \cite{ballato2008silicon}, \cite{girard2019overview}.

\begin{figure}[h]
    \centering
    \begin{subfigure}[b]{0.4\textwidth}
        \centering
        \includegraphics[width=\textwidth]{plots/sio4.png}
       
    \end{subfigure}
    \begin{subfigure}[b]{0.4\textwidth}
        \centering
        \includegraphics[width=\textwidth]{plots/etrap.png}
    \end{subfigure}
    \caption{\label{fig:asio2}(a) SiO$_4$ tetrahedron in amorphous silicon dioxide, visualized in vmd \cite{HUMP96}.(b) An intrinsic electron trap in a-SiO2 with spin distribution \cite{idintrinsic}.}
\end{figure}

Defects are of interest when utilizing amorphous solids in devices, as they can completely break down due to unwanted defects, while others may depend on the creation and migration of defects \cite{phung2020role}, \cite{idintrinsic}. There are different types of defects that can manifest in amorphous silica, among them are the peroxy bridge, O-O bonds, NBOHC (dangling-bond), hydroxyl-E' centres, oxygen vacancies (both relaxed and non-relaxed) and electron traps \cite{skuja}. The focus of this thesis is the manifestation of spontaneous intrinsic electron traps in amorphous silicon dioxide. An electron can become trapped spontaneously deep in the bandgap by coming too close to a trapping site which intrinsically exists in the structure. This process fills the electronic state, which leads to structural distortion and localization of an additional electron on a Si atom \cite{atomisticmod}. The structural change can manifest as an opening of the angles centred on the Si atom and elongation of the surrounding bonds. Figure \ref{fig:asio2}(b) is a visualization of an electron trap from \cite{idintrinsic}. The spin density in Figure \ref{fig:asio2}(b) is the spin density of the trapped electron which appears as lobes on the atoms. These types of defects are difficult to identify without costly calculations since the trapping sites can be seemingly indistinguishable from normal sites. 

The variations of the structure make it difficult to accurately fingerprint defects by using simple approximations such as angle or bond length cut-offs. The go-to way of determining whether a site in the solid can trap an electron is by performing costly ab-initio calculations. This project aims to utilize machine learning to predict the presence of defects based on complex fingerprints. Machine learning has become more prominent in physics as it is capable of learning from massive amounts of data, and can compute properties faster with a high level of accuracy compared to conventional methods. Machine learning requires representative input data which have been constructed to portray the learning cases through properties which are related to the target class. Additionally, most model predictions are highly dependent on the optimization process for good predictive performance. The machine learning models used are the gradient boosted decision trees, XGBoost and Catboost. A descriptor consisting of properties such as bond lengths, angles, dihedrals and ring structures will be compared to DScribe's atom-centred symmetry functions. 


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

