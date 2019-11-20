'''
The Python Wrapper for the Robust Peak peakFinder
Authors: Marjan Hadian Jazi
	 Alireza Sadri

Inputs:
inData : This is the 2D input image as a numpy 2d-array.
inMask : This is the bad pixel mask.
		default: 1 for all pixels
LAMBDA : The ratio of a Guassian Profile over its standard deviation that is assumed as inlier
		default: 4 Sigma (Sigma being its STD)
SNR_ACCEPT: Traditionally, SNR is one of the factors to reject bad peakListCheetah
		default: 8.0
PEAK_MAX_PIX: maximum number of pixels in a peak.
		default: 50
PEAK_MIN_PIX: Minimum number of pixels in a peak.
		default: 50		

Output:
peakListCheetah is a numpy 2D-array in the style of Cheetah's output.
Rows are peaks and coloums are:
-------------------------------------------------------------------------
Mass_Center_X, Mass_Center_Y, Mass_Total, Number of pixels in a peak, Maximum value, SNR
-------------------------------------------------------------------------

You can get the number of peaks by YOUROUTPUTNAME.shape[0]
'''

import numpy
import ctypes
peakFinderPythonLib = ctypes.cdll.LoadLibrary("/home/sadriali/Projects/RPFOrig/Debug/libRPFOrig.so")
peakFinderPythonLib.peakFinder.restype = ctypes.c_int
peakFinderPythonLib.peakFinder.argtypes = [
				ctypes.c_double, ctypes.c_double,
				numpy.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
				numpy.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                numpy.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]


def robustPeakFinderPyFunc(inData, inMask = None,
				LAMBDA = 4.0,
				SNR_ACCEPT = 8.0,
				PEAK_MAX_PIX = 50,
				PEAK_MIN_PIX = 1):
    inData = numpy.double(inData)
    if(inMask is None):
        inMask = 1 + 0*inData
    else:
        inMask = numpy.double(inMask)
    peakListCheetah = numpy.zeros([1024, 6])
    szx, szy = inData.shape
    peak_cnt = peakFinderPythonLib.peakFinder(LAMBDA, SNR_ACCEPT,
						inData, inMask, szy, szx,
						PEAK_MAX_PIX, PEAK_MIN_PIX, peakListCheetah)
    return peakListCheetah[:peak_cnt]
