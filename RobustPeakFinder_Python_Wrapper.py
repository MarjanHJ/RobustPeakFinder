'''
The Python Wrapper for the Robust Peak peakFinder
Authors: Marjan Hadian Jazi
		 Alireza Sadri

Inputs:
indata : This is the 2D input image as a numpy 2d-array.
LAMBDA : The ratio of a Guassian Profile over its standard deviation that is assumed as inlier
		default: 4 Sigma (Sigma being its STD)
SNR_ACCEPT: Traditionally, SNR is one of the factors to reject bad peakListCheeta
		default: 8.0
PEAK_MAX_PIX: Number of pixels in a peak.
		default: 50

Output:
peakListCheeta is a numpy 2D-array in the style of Cheetah's output.
Rows are peaks and coloums are:
-------------------------------------------------------------------------
Mass_Center_X, Mass_Center_Y, Mass_Total, Number of pixels in a peak
-------------------------------------------------------------------------

You can get the number of peaks by peakListCheeta.shape()[0]
'''

import numpy
import ctypes
peakFinderPythonLib = ctypes.cdll.LoadLibrary("./RobustPeakFinder.so")
peakFinderPythonLib.peakFinder.restype = ctypes.c_int
peakFinderPythonLib.peakFinder.argtypes = [
				ctypes.c_double, ctypes.c_double,
				numpy.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                numpy.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]


def robustPeakFinderPyFunc(indata,
				LAMBDA = 4.0,
				SNR_ACCEPT = 8.0,
				PEAK_MAX_PIX = 50):
    peakListCheeta = numpy.zeros([50000, 4])
    szx, szxy = indata.shape
    peak_cnt = peakFinderPythonLib.peakFinder(LAMBDA, SNR_ACCEPT,
						indata, szxy, szx,
						PEAK_MAX_PIX, peakListCheeta)
    return peakListCheeta[:peak_cnt]
