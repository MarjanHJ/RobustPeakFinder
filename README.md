# RobustPeakFinder
A C Library for Robust Peak Finding in 2D images with Python/MATLAB Wrapper

# Introduction
The RobustPeakFinder.c is writetn in C language and can be used to detect Bragg peaks in a 2D diffraction pattern or to count peaks such as stars .... The relevant paper to cite can be found here:

# compilation into shared library
Run the following command:

```
g++ -fPIC -shared -o RobustPeakFinder.so RobustPeakFinder.c
```

Tha wrapper will be looking for this shared library.

# Usage from Python
A Python wrapper is written in the file RobustPeakFinder_Python_Wrapper.py

# Example in Python 
A Python test is also provided in the file RobustPeakFinder_Python_Test_for_AS.py for Australian Synchotron
