# RobustPeakFinder
A C Library for Robust Peak Finding in 2D images with Python/MATLAB Wrapper

# Introduction
The RobustPeakFinder.c is writetn in C language and can be used to detect Bragg peaks in a 2D diffraction pattern or to count peaks  in images such as stars. The relevant paper to cite can be found here:

# compilation into shared library
Run the following command to generate a shared library .so:
```
make
```

# Usage from Python
A Python wrapper is written in the file RobustPeakFinder_Python_Wrapper.py. Tha wrapper will be looking for the .so shared library file.

# Examples in Python 
Two Python tests are also provided:

1- A Test written specifically to read one image from a HDF5 file made in Australian Synchotron in the file: RobustPeakFinder_Python_Test_for_AS.py

2- A Simple test to prove that the code is working in the file: RobustPeakFinder_Python_Test_Simple.py. Simply type in:
```
make test
```
