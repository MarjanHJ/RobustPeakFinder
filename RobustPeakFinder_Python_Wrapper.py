import numpy as np
import ctypes
import os
from robustLib.textProgBar import textProgBar
from multiprocessing import Process, Queue, cpu_count

dir_path = os.path.dirname(os.path.realpath(__file__))
peakFinderPythonLib = ctypes.cdll.LoadLibrary(dir_path + '/RobustPeakFinder.so')
                
'''
int peakFinder(	float *inData, float *inMask,
				float *minPeakValMap, float *maxBackMeanMap, 
				float *peakList, int MAXIMUM_NUMBER_OF_PEAKS,
				float bckSNR, float pixPAPR,
				int XPIX, int YPIX, int PTCHSZ,	
				int PEAK_MIN_PIX, int PEAK_MAX_PIX)
'''
peakFinderPythonLib.peakFinder.restype = ctypes.c_int
peakFinderPythonLib.peakFinder.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_int,
                ctypes.c_float, ctypes.c_float,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                ctypes.c_int, ctypes.c_int]


def robustPeakFinderPyFunc(inData, 
                inMask = None,
                minPeakValMap = None, 
                maxBackMeanMap = None, 
                bckSNR = 6.0,
                pixPAPR = 2.0,
                PTCHSZ = 16,
                PEAK_MAX_PIX = 25,
                PEAK_MIN_PIX = 1,
                MAXIMUM_NUMBER_OF_PEAKS = 1024):
    """
    The Python Wrapper for the Robust Peak peakFinder
    Authors: Marjan Hadian Jazi
             Alireza Sadri

    Inputs:
    inData : This is the 2D input image as a np 2d-array.
    inMask : This is the bad pixel mask.
            default: 1 for all pixels, with the size of inData
    minPeakValMap : a threshold for peak maximum value
            default: 0 with the size of inData
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
    peakListCheetah is a np 2D-array in the style of Cheetah's output for CXI files
    Each rows is for each peak and coloums are:
    -----------------------------------------------------------------------------------------
    Mass_Center_X, Mass_Center_Y, Mass_Total, Number of pixels in a peak, Maximum value, SNR
    -----------------------------------------------------------------------------------------

    You can get the number of peaks by YOUROUTPUTNAME.shape[0]
    """    
    inDataShape = inData.shape
    if(inMask is None):
        inMask = np.ones(inDataShape, dtype='uint8')
    else:
        inMask = inMask.astype('uint8')
    if(maxBackMeanMap is None):
        maxBackMeanMap = np.ones(inDataShape, dtype='float32')*np.finfo('float32').max
    else:
        maxBackMeanMap = maxBackMeanMap.astype('float32')
    if(minPeakValMap is None):
        minPeakValMap = np.zeros(inDataShape, dtype='float32')
    else:
        minPeakValMap = minPeakValMap.astype('float32')
    
    inData = inData.astype('float32')
    
    peakListCheetah = np.zeros([MAXIMUM_NUMBER_OF_PEAKS, 6], dtype='float32')    

    peak_cnt = peakFinderPythonLib.peakFinder(  inData.flatten('F'), 
                                                inMask.flatten('F'), 
                                                minPeakValMap.flatten('F'), 
                                                maxBackMeanMap.flatten('F'),
                                                peakListCheetah,
                                                MAXIMUM_NUMBER_OF_PEAKS,
                                                bckSNR, pixPAPR,
                                                inDataShape[0], 
                                                inDataShape[1], 
                                                PTCHSZ, 
                                                PEAK_MIN_PIX, 
                                                PEAK_MAX_PIX)

    return peakListCheetah[:peak_cnt]
    
def robustPeakFinderPyFunc_multiprocFunc(queue, baseImgCnt, 
                                                inDataT, 
                                                inMaskT,
                                                minPeakValMapT,
                                                maxBackMeanMapT,
                                                bckSNR,
                                                pixPAPR,
                                                PTCHSZ,
                                                PEAK_MAX_PIX,
                                                PEAK_MIN_PIX,
                                                MAXIMUM_NUMBER_OF_PEAKS):
    for imgCnt in range(inDataT.shape[0]):
        peakList = robustPeakFinderPyFunc(  inData = inDataT[imgCnt],
                                            inMask = inMaskT[imgCnt],
                                            minPeakValMap = minPeakValMapT[imgCnt], 
                                            maxBackMeanMap = maxBackMeanMapT[imgCnt],
                                            bckSNR = bckSNR,
                                            pixPAPR = pixPAPR,
                                            PTCHSZ = PTCHSZ,
                                            PEAK_MAX_PIX = PEAK_MAX_PIX,
                                            PEAK_MIN_PIX = PEAK_MIN_PIX,
                                            MAXIMUM_NUMBER_OF_PEAKS = MAXIMUM_NUMBER_OF_PEAKS)
        queue.put(list([baseImgCnt+imgCnt, peakList]))
    
def robustPeakFinderPyFunc_multiproc(   inData, 
                                        inMask = None,
                                        minPeakValMap = None, 
                                        maxBackMeanMap = None, 
                                        bckSNR = 6.0,
                                        pixPAPR = 2.0,
                                        PTCHSZ = 16,
                                        PEAK_MAX_PIX = 25,
                                        PEAK_MIN_PIX = 1,
                                        MAXIMUM_NUMBER_OF_PEAKS = 1024):
    inDataShape = inData.shape
    if(inMask is None):
        inMask = np.ones(inDataShape, dtype='uint8')
    if(maxBackMeanMap is None):
        maxBackMeanMap = np.ones(inDataShape, dtype='float32')*np.finfo('float32').max
    if(minPeakValMap is None):
        minPeakValMap = np.zeros(inDataShape, dtype='float32')
        
    f_N = inData.shape[0]
    
    peakListTensor = np.zeros((f_N, 1024, 6), dtype='float32')
    nPeaks = np.zeros(f_N, dtype='uint32')
    
    queue = Queue()
    mycpucount = cpu_count() - 1
    print('Multiprocessing ' + str(f_N) + ' frames...')
    numProc = f_N
    numSubmitted = 0
    numProcessed = 0
    numBusyCores = 0
    firstProcessed = 0
    default_stride = np.maximum(mycpucount, int(numProc/mycpucount/2))
    while(numProcessed<numProc):
        if (not queue.empty()):
            qElement = queue.get()
            _imgCnt = qElement[0]
            _tmpResult = qElement[1]
            nPeaks[_imgCnt] = _tmpResult.shape[0]
            peakListTensor[_imgCnt, :nPeaks[_imgCnt], :] = _tmpResult
            numBusyCores -= 1
            numProcessed += 1
            if(firstProcessed==0):
                pBar = textProgBar(numProc-1, title = 'RPF progress')
                firstProcessed = 1
            else:
                pBar.go(1)
            continue;

        if((numSubmitted<numProc) & (numBusyCores < mycpucount)):
            baseImgCnt = numSubmitted
            stride = np.minimum(default_stride, numProc - numSubmitted)
            if(stride==1):
                baseImgCnt -= 1
            Process(target = robustPeakFinderPyFunc_multiprocFunc, 
                    args=((queue, baseImgCnt,
                            inData[baseImgCnt:baseImgCnt+stride], 
                            inMask[baseImgCnt:baseImgCnt+stride],
                            minPeakValMap[baseImgCnt:baseImgCnt+stride],
                            maxBackMeanMap[baseImgCnt:baseImgCnt+stride],
                            bckSNR, 
                            pixPAPR,
                            PTCHSZ,
                            PEAK_MAX_PIX,
                            PEAK_MIN_PIX,
                            MAXIMUM_NUMBER_OF_PEAKS))).start()
            numSubmitted += stride
            numBusyCores += 1
    del pBar
    return(nPeaks, peakListTensor)
