# RobustPeakFinder
A C Library for Robust Peak Finding in 2D images with Python Wrapper. The relevant paper can be found [here](http://scripts.iucr.org/cgi-bin/paper?S1600576717014340)

# Introduction
The RobustPeakFinder.c is writetn in C language and can be used to detect Bragg peaks in a 2D diffraction pattern or to count peaks in any other image. 

# A Note on the Bad Pixel Mask
For users in EuropeanXFEL (SPB and MID) - which use 1Mp AGIPD detectors -, since the calibration ignores the variance in Analog values, they may end up as negative numbers. Statistically, it is still a valid number and must be considered in the analysis. However, this usually happens to bad pixels. As such giving the bad pixel mask in the input is necessary.

# Compilation into shared library
Run the following command to generate a shared library RobustPeakFinder.so:
```
make
```
**Note**: using the first line of the C file also does the task.
# Usage from Python
A Python wrapper is written in the file RobustPeakFinder_Python_Wrapper.py. Tha wrapper will be looking for the .so shared library file.

# Examples in Python 
Two Python tests are also provided:

1- A Test written specifically to read a HDF5 file from Australian Synchotron:
RobustPeakFinder_Python_Test_for_AS.py

2- A Simple test to prove that the code is working in the file: RobustPeakFinder_Python_Test_Simple.py.
Simply type in:
```
make test
```
