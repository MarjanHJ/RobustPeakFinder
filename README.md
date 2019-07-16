# RobustPeakFinder
A C Library for Robust Peak Finding in 2D images with Python/MATLAB Wrapper. The relevant paper can be found [here](http://scripts.iucr.org/cgi-bin/paper?S1600576717014340)

# Introduction
The RobustPeakFinder.c is writetn in C language and can be used to detect Bragg peaks in a 2D diffraction pattern or to count peaks in any other image. 

# Compilation into shared library
Run the following command to generate a shared library .so:
```
make
```
**Note**: Uncomenting the first line of the C file also do the task.
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

# European XFEL example:
This must be run on the maxwell cluster in order to access the example data
Usage (on maxwell)
```
make
module load anaconda3
python RobustPeakFinder_Python_Test_for_EuXFEL.py
```

