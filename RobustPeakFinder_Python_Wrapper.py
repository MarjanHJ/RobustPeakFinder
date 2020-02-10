'''
The Python Wrapper for the Robust Peak peakFinder
Authors: Marjan Hadian Jazi
     Alireza Sadri

Inputs:
inData : This is the 2D input image as a numpy 2d-array.
inMask : This is the bad pixel mask.
        default: 1 for all pixels, with the size of inData
maxBiasMap : maximum value of the model for the background
        default: 32000 with the size of inData
thresholdMap : a threshold that will be used added to the estimated model for the background 
        default: 0 with the size of inData
AbProbMap : an abnormality probability that is multplied to the SNR and also used to preform weighted regression for background
        default: 1 with the size of inData
StdsMap : standard deviation of variations of pixels in the background used in calculation of SNR
        default: 0 with the size of inData
LAMBDA : The ratio of a Guassian Profile over its standard deviation that is assumed as inlier
        default: 4 Sigma (Sigma being its STD)
SNR_ACCEPT: Traditionally, SNR is one of the factors to reject bad peakListCheetah
        default: 8.0
PEAK_MAX_PIX: maximum number of pixels in a peak.
        default: 50
PEAK_MIN_PIX: Minimum number of pixels in a peak.
        default: 1        
MAXIMUM_NUMBER_OF_PEAKS: self explanatory
        default = 1024
        
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
peakFinderPythonLib = ctypes.cdll.LoadLibrary("RobustPeakFinder.so")
peakFinderPythonLib.peakFinder.restype = ctypes.c_int
peakFinderPythonLib.peakFinder.argtypes = [
                ctypes.c_double, ctypes.c_double,
                numpy.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                numpy.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                numpy.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                numpy.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                numpy.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                numpy.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                numpy.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]


def robustPeakFinderPyFunc(inData, 
                maxBiasMap = None, 
                thresholdMap = None, 
                AbProbMap = None, 
                StdsMap = None, 
                inMask = None,
                LAMBDA = 4.0,
                SNR_ACCEPT = 8.0,
                PEAK_MAX_PIX = 25,
                PEAK_MIN_PIX = 1,
                MAXIMUM_NUMBER_OF_PEAKS = 1024):
    inData = numpy.double(inData)
    if(maxBiasMap is None):
        maxBiasMap = 0*inData.copy() + 32000
    else:
        maxBiasMap = numpy.double(maxBiasMap)
    if(thresholdMap is None):
        thresholdMap = 0*inData.copy()
    else:
        thresholdMap = numpy.double(thresholdMap)
    if(inMask is None):
        inMask = 1 + 0*inData.copy()
    else:
        inMask = numpy.double(inMask)
    if(AbProbMap is None):
        AbProbMap = 1 + 0*inData.copy()
    else:
        AbProbMap = numpy.double(AbProbMap)
    if(StdsMap is None):
        StdsMap = 0*inData.copy()
    else:
        StdsMap = numpy.double(StdsMap)
    
    #if you change the maximum numbre of peaks here, the C code must know about it.    
    peakListCheetah = numpy.zeros([MAXIMUM_NUMBER_OF_PEAKS, 6])    
    szx, szy = inData.shape
    peak_cnt = peakFinderPythonLib.peakFinder(LAMBDA, SNR_ACCEPT,
                        inData, AbProbMap, StdsMap, inMask, thresholdMap, maxBiasMap,
                        szy, szx, PEAK_MAX_PIX, PEAK_MIN_PIX, MAXIMUM_NUMBER_OF_PEAKS,
                        peakListCheetah)
    return peakListCheetah[:peak_cnt]
