# RobustPeakFinder
A C Library for Robust Peak Finding in 2D images with Python Wrapper. The relevant paper can be found [here](http://scripts.iucr.org/cgi-bin/paper?S1600576717014340)

## Introduction
The RobustPeakFinder.c is writetn in C language and can be used to detect Bragg peaks in a 2D diffraction pattern or to count peaks in any other image. The method is based on the library implemented in RGFLib.c presented in [Robust Gaussian Fitting library](https://github.com/ARSadri/RGFLib). A modification of CrystFEL is now availble that performs RPF to a good extent and can be found in [robustFEL](https://stash.desy.de/projects/RFEL) project.

### A Note on the Bad Pixel Mask
Giving the bad pixel mask in the input is necessary. The RPF algorithm is looking for outstanding pixels. If bad pixels who are outstanding often, are not masked, they will be picked up. We suggest to use the mask makers found in [robustFEL](https://stash.desy.de/projects/RFEL) project.

## Compilation into shared library
After cloning this repo and putting the [RGFLib.c](https://raw.githubusercontent.com/ARSadri/RobustGaussianFittingLibrary/master/RobustGaussianFittingLibrary/RGFLib.c) beside this Makefile, run the following command to generate a shared library RobustPeakFinder.so:
```
make
```
**Note**: using the first line of the C file also compiles the library.
## Usage from Python
A Python wrapper is written in the file RobustPeakFinder_Python_Wrapper.py. Tha wrapper will be looking for the .so shared library file.

### Examples in Python 
Two Python tests are also provided:

1- A Test written specifically to read a HDF5 file from Australian Synchotron:
RobustPeakFinder_Python_Test_for_AS.py

2- A Simple test to prove that the code is working in the file: RobustPeakFinder_Python_Test_Simple.py.
Simply type in:
```
make test
```
