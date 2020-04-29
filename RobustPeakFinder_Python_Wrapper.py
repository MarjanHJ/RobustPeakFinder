"""
The Python Wrapper for the Robust Peak peakFinder
Authors: Marjan Hadian Jazi
         Alireza Sadri

Inputs:
inData : This is the 2D input image as a numpy 2d-array.
inMask : This is the bad pixel mask.
        default: 1 for all pixels, with the size of inData
minPeakValMap : a threshold for peak maximum value
        default: 0 with the size of inData
SNRFactor : will be multiplied to SNR, useful if badpixel mask generates probabilities
        default: 1 with the size of inData
maxBackMeanMap : maximum value of the model for the background, 
                   useful for stopping background to go to nonlinear regions
        default: 32000 with the size of inData
peakList : the output array of size MAXIMUM_NUMBER_OF_PEAKS x 6
        default: all zeros
MAXIMUM_NUMBER_OF_PEAKS: self explanatory
        default = 1024
bckSNR: Background SNR
        default: 6.0
pixPAPR : pixels Peak to average power ratio
        default: 2.0
PEAK_MIN_PIX: Minimum number of pixels in a peak.
        default: 1        
PEAK_MAX_PIX: maximum number of pixels in a peak.
        default: 25
        
Output:
peakListCheetah is a numpy 2D-array in the style of Cheetah's output for CXI files
Each rows is for each peak and coloums are:
-----------------------------------------------------------------------------------------
Mass_Center_X, Mass_Center_Y, Mass_Total, Number of pixels in a peak, Maximum value, SNR
-----------------------------------------------------------------------------------------

You can get the number of peaks by YOUROUTPUTNAME.shape[0]
"""

import numpy
import ctypes
peakFinderPythonLib = ctypes.cdll.LoadLibrary("./RobustPeakFinder.so")
                
'''
int peakFinder(	float *inData, float *inMask,
				float *SNRFactor, float *minPeakValMap, float *maxBackMeanMap, 
				float *peakList, int MAXIMUM_NUMBER_OF_PEAKS,
				float bckSNR, float pixPAPR,
				int XPIX, int YPIX, int PTCHSZ,	
				int PEAK_MIN_PIX, int PEAK_MAX_PIX)
'''
peakFinderPythonLib.peakFinder.restype = ctypes.c_int
peakFinderPythonLib.peakFinder.argtypes = [
                numpy.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                numpy.ctypeslib.ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"),
                numpy.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                numpy.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                numpy.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                numpy.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_int,
                ctypes.c_float, ctypes.c_float,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]


def robustPeakFinderPyFunc(inData, 
                inMask = None,
                minPeakValMap = None, 
                maxBackMeanMap = None, 
                SNRFactor = None, 
                bckSNR = 6.0,
                pixPAPR = 2.0,
                PTCHSZ = 16,
                PEAK_MAX_PIX = 25,
                PEAK_MIN_PIX = 1,
                MAXIMUM_NUMBER_OF_PEAKS = 1024):
    
    inDataSize = inData.size
    XPIX, YPIX = inData.shape
    if(inMask is None):
        inMask = numpy.ones(inDataSize, dtype='uint8')
    else:
        inMask = inMask.astype('uint8')
    if(maxBackMeanMap is None):
        maxBackMeanMap = numpy.ones(inDataSize, dtype='float32')*numpy.finfo('float32').max
    else:
        maxBackMeanMap = maxBackMeanMap.astype('float32')
    if(minPeakValMap is None):
        minPeakValMap = numpy.zeros(inDataSize, dtype='float32')
    else:
        minPeakValMap = minPeakValMap.astype('float32')
    if(SNRFactor is None):
        SNRFactor = numpy.ones(inDataSize, dtype='float32')
    else:
        SNRFactor = SNRFactor.astype('float32')
    
    inData = inData.astype('float32')
    
    peakListCheetah = numpy.zeros([MAXIMUM_NUMBER_OF_PEAKS, 6], dtype='float32')    
    
    peak_cnt = peakFinderPythonLib.peakFinder(inData.flatten('F'), inMask.flatten('F'), 
                        SNRFactor.flatten('F'), minPeakValMap.flatten('F'), maxBackMeanMap.flatten('F'),
                        peakListCheetah, MAXIMUM_NUMBER_OF_PEAKS,
                        bckSNR, pixPAPR,
                        XPIX, YPIX, PTCHSZ, PEAK_MIN_PIX, PEAK_MAX_PIX)

    peakListCheetah[:, 0], peakListCheetah[:, 1] = peakListCheetah[:, 1], peakListCheetah[:, 0].copy()                    
                        
    return peakListCheetah[:peak_cnt]
