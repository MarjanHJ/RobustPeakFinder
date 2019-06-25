# RobustPeakFinder
A C Library for Robust Peak Finding in 2D images with Python/MATLAB Wrapper

# Introduction
The RobustPeakFinder.c is writetn in C language and can be used to detect Bragg peaks in a 2D diffraction pattern or to count peaks such as stars .... The relevant paper to cite can be found here:

# compilation into shared library
Run the following command:

```
g++ -fPIC -shared -o RobustPeakFinder.so RobustPeakFinder.c
```

Tha wrapper will be looking for "RobustPeakFinder.so". if you changed this name, you have to change it in the wrapper too.

# Usage from Python
A Python wrapper is written in the file RobustPeakFinder_Python_Wrapper.py

# Example in Python
A Python test is also provided in the file RobustPeakFinder_Python_Test.py
The example file includes two tests, first one generates a synthetic image with a few peaks in it to detect. The second test reads the image star.jpg and count stars that are reletively small. To detect larger stars, one hase to play with user parameters in the wrapper or the C file.

# Usage from MATLAB
TBA
