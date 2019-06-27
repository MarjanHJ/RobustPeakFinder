# RobustPeakFinder
A C Library for Robust Peak Finding in 2D images with Python/MATLAB Wrapper

# Introduction
The RobustPeakFinder.c is writetn in C language and can be used to detect Bragg peaks in a 2D diffraction pattern or to count peaks  in images such as stars. The relevant paper can be found [here](http://scripts.iucr.org/cgi-bin/paper?S1600576717014340)

# Compilation into shared library
Run the following command to generate a shared library .so:
```
make
```
**Note**: the first comment line in the C file can also fo the task.
# Usage from Python
A Python wrapper is written in the file RobustPeakFinder_Python_Wrapper.py. Tha wrapper will be looking for the .so shared library file.

### Inputs:
* **inData**: This is the 2D input image as a numpy 2d-array.
* **inMask**: This is the bad pixel mask.
		default: 1 for all pixels
* **LAMBDA**: The ratio of a Guassian Profile over its standard deviation that is assumed as inlier
		default: 4 Sigma (Sigma being its STD)
* **SNR_ACCEPT**: Traditionally, SNR is one of the factors to reject bad peakListCheeta
		default: 8.0
* **PEAK_MAX_PIX**: Number of pixels in a peak.
		default: 50

### Output:
The function's output is a numpy 2D-array in the style of Cheetah's output.
Rows are for each peak and coloums are:

| First Moment X | First Moment Y | Sum of all values | Number of pixels in a peak |
| -------------- | -------------- | ----------------- | -------------------------- |

You can get the number of peaks by YOUR_OUTPUT_VARIABLE_NAME.shape()[0]

# Examples in Python 
Two Python tests are also provided:

1- A Test written specifically to read a HDF5 file from Australian Synchotron:
RobustPeakFinder_Python_Test_for_AS.py

2- A Simple test to prove that the code is working in the file: RobustPeakFinder_Python_Test.py.
Simply type in:
```
make test
```
