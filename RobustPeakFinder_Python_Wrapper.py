import numpy as np
import ctypes
import os
from multiprocessing import Process, Queue, cpu_count

############ if you are going to use cython, just change this part #############
dir_path = os.path.dirname(os.path.realpath(__file__))
peakFinderPythonLib = ctypes.cdll.LoadLibrary(dir_path + '/RobustPeakFinder.so')
""""RPF main calling function:
int peakFinder( float *inData, unsigned char *inMask, unsigned char *peakMask, 
                float *darkThreshold, float *singlePhotonADU, 
                float *maxBackMeanMap, float *peakList, float *peakMap,
                int MAXIMUM_NUMBER_OF_PEAKS,
                float bckSNR, float pixPAPR,
                int XPIX, int YPIX, int PTCHSZ,    
                int PEAK_MIN_PIX, int PEAK_MAX_PIX, 
                int optIters, int finiteSampleBias))
"""
peakFinderPythonLib.peakFinder.restype = ctypes.c_int
peakFinderPythonLib.peakFinder.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_float,
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_int,
                ctypes.c_float, ctypes.c_float,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
################################################################################

def robustPeakFinderPyFunc(inData, 
                inMask = None,
                peakMask = None,
                darkThreshold = None,
                singlePhotonADU = 1,
                maxBackMeanMap = None, 
                bckSNR = 6.0,
                pixPAPR = 2.0,
                PTCHSZ = 25,
                PEAK_MAX_PIX = 25,
                PEAK_MIN_PIX = 1,
                MAXIMUM_NUMBER_OF_PEAKS = 1024,
                optIters = 8,
                finiteSampleBias = 200,
                returnPeakMap = False):
    """Robust Peak Finder parameters
    The Python Wrapper for the Robust Peak peakFinder
    Authors: Marjan Hadian Jazi
             Alireza Sadri

    Inputs:
    inData : This is the 2D input image as a np 2d-array.
    inMask : This is the bad pixel mask.
            default: 1 for all pixels, with the size of inData
    peakMask : given a peak mask made by peakNet, it will not look for peaks if peakMask is 0
            default: 1 for all pixels, with the size of inData
    darkThreshold : give darkSNR * STD of the values of pixels in the dark
            darkSNR = 4 to 6 gives good results.
            default: 0 with the size of inData
    singlePhotonADU : relate pixel values to variance of background under high intensity Flat field
                    The slope of the line passing through zero gives you the single photon ADU
                    NOTE: using low intensity flat field will not allow you model the nonlinearities
                    of the system with your line.
            default: 1
    maxBackMeanMap : maximum value of the model for the background, 
                       useful for stopping background to go to nonlinear regions
            default: np.fino('float32').max with the size of inData
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
    optIters: Number of iterations of order stantistics optimization (FLKOS)
            default: 10
    Output:
    peakList is a np 2D-array in the style of Cheetah's output for CXI files
    Each row is for each peak and coloums are:
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
    if(peakMask is None):
        peakMask = inMask.copy()
    else:
        peakMask = (peakMask*inMask.copy()).astype('uint8')
    if(maxBackMeanMap is None):
        maxBackMeanMap = np.ones(inDataShape, dtype='float32')*np.finfo('float32').max
    else:
        maxBackMeanMap = maxBackMeanMap.astype('float32')
    if(darkThreshold is None):
        darkThreshold = np.zeros(inDataShape, dtype='float32')
    else:
        darkThreshold = darkThreshold.astype('float32')

    inData = inData.astype('float32')
    
    peakList = np.zeros([MAXIMUM_NUMBER_OF_PEAKS, 6], dtype='float32')    
    if(returnPeakMap):
        peakMap = np.zeros(shape = inData.shape, dtype = inData.dtype)
    else:
        peakMap = np.ones(shape = (2,2), dtype = inData.dtype)
    peakMap = peakMap.flatten('F')
    
    peak_cnt = peakFinderPythonLib.peakFinder(inData.flatten('F'), 
                                              inMask.flatten('F'), 
                                              peakMask.flatten('F'),
                                              darkThreshold.flatten('F'),
                                              singlePhotonADU,
                                              maxBackMeanMap.flatten('F'),
                                              peakList,
                                              peakMap,
                                              MAXIMUM_NUMBER_OF_PEAKS,
                                              bckSNR, pixPAPR,
                                              inDataShape[0], 
                                              inDataShape[1], 
                                              PTCHSZ, 
                                              PEAK_MIN_PIX, 
                                              PEAK_MAX_PIX,
                                              optIters,
                                              finiteSampleBias)
    if(returnPeakMap):
        return((peakList[:peak_cnt], peakMap.reshape(inData.shape[1], inData.shape[0]).T))
    else:
         return(peakList[:peak_cnt])
    
def robustPeakFinderPyFunc_multiprocFunc(queue, 
                                        baseImgCnt, 
                                        inDataT, 
                                        inMaskT,
                                        peakMaskT,
                                        darkThreshold,
                                        singlePhotonADU,
                                        backgroundMaxT,
                                        bckSNR,
                                        pixPAPR,
                                        PTCHSZ,
                                        PEAK_MAX_PIX,
                                        PEAK_MIN_PIX,
                                        MAXIMUM_NUMBER_OF_PEAKS,
                                        optIters,
                                        finiteSampleBias,
                                        returnPeakMap):
    for imgCnt in range(inDataT.shape[0]):
        toUnpack = robustPeakFinderPyFunc( inData = inDataT[imgCnt],
                                            inMask = inMaskT[imgCnt],
                                            peakMask = peakMaskT[imgCnt],
                                            darkThreshold = darkThreshold[imgCnt], 
                                            singlePhotonADU = singlePhotonADU, 
                                            maxBackMeanMap = backgroundMaxT[imgCnt],
                                            bckSNR = bckSNR,
                                            pixPAPR = pixPAPR,
                                            PTCHSZ = PTCHSZ,
                                            PEAK_MAX_PIX = PEAK_MAX_PIX,
                                            PEAK_MIN_PIX = PEAK_MIN_PIX,
                                            MAXIMUM_NUMBER_OF_PEAKS = MAXIMUM_NUMBER_OF_PEAKS,
                                            optIters = optIters,
                                            finiteSampleBias = finiteSampleBias,
                                            returnPeakMap = returnPeakMap)
        if(returnPeakMap):
            peakList, peakMap = toUnpack
            queue.put(list([baseImgCnt+imgCnt, peakList, peakMap]))
        else:
            peakList = toUnpack
            queue.put(list([baseImgCnt+imgCnt, peakList]))

def robustPeakFinderPyFunc_multiproc(inData, 
                                        inMask = None,
                                        peakMask = None,
                                        darkThreshold = None, 
                                        singlePhotonADU = 1,
                                        maxBackMeanMap = None, 
                                        bckSNR = 6.0,
                                        pixPAPR = 2.0,
                                        PTCHSZ = 16,
                                        PEAK_MAX_PIX = 25,
                                        PEAK_MIN_PIX = 1,
                                        MAXIMUM_NUMBER_OF_PEAKS = 1024,
                                        optIters = 10,
                                        finiteSampleBias = 200,
                                        returnPeakMap = False):
    inDataShape = inData.shape
    if(inMask is None):
        inMask = np.ones(inDataShape, dtype='uint8')
    if(peakMask is None):
        peakMask = inMask.copy()
    if(maxBackMeanMap is None):
        maxBackMeanMap = np.ones(inDataShape, dtype='float32')*np.finfo('float32').max
    if(darkThreshold is None):
        darkThreshold = np.zeros(inDataShape, dtype='float32')

    f_N = inData.shape[0]
    
    peakListTensor = np.zeros((f_N, 1024, 6), dtype='float32')
    if(returnPeakMap):
        peakMapTensor = np.zeros(shape = inDataShape, dtype = 'float32')
    nPeaks = np.zeros(f_N, dtype='uint32')
    
    queue = Queue()
    mycpucount = cpu_count()
    print('Multiprocessing ' + str(f_N) + ' frames with ' + str(mycpucount) + ' CPUs...')
    numProc = f_N
    numSubmitted = 0
    numProcessed = 0
    numBusyCores = 0
    default_stride = np.maximum(mycpucount, int(numProc/mycpucount/2))
    while(numProcessed<numProc):
        if (not queue.empty()):
            qElement = queue.get()
            _imgCnt = qElement[0]
            _tmpResult = qElement[1]
            nPeaks[_imgCnt] = _tmpResult.shape[0]
            peakListTensor[_imgCnt, :nPeaks[_imgCnt], :] = _tmpResult
            if(returnPeakMap):
                peakMapTensor[_imgCnt] = qElement[2]
            
            numBusyCores -= 1
            numProcessed += 1
            continue;   #its allways good to empty the queue

        if((numSubmitted<numProc) & (numBusyCores < mycpucount)):
            baseImgCnt = numSubmitted
            stride = np.minimum(default_stride, numProc - numSubmitted)
            if(stride==1):
                baseImgCnt -= 1                
            Process(target = robustPeakFinderPyFunc_multiprocFunc, 
                    args=((queue, baseImgCnt,
                            inData[baseImgCnt:baseImgCnt+stride], 
                            inMask[baseImgCnt:baseImgCnt+stride],
                            peakMask[baseImgCnt:baseImgCnt+stride],
                            darkThreshold[baseImgCnt:baseImgCnt+stride],
                            singlePhotonADU,
                            maxBackMeanMap[baseImgCnt:baseImgCnt+stride],
                            bckSNR, 
                            pixPAPR,
                            PTCHSZ,
                            PEAK_MAX_PIX,
                            PEAK_MIN_PIX,
                            MAXIMUM_NUMBER_OF_PEAKS,
                            optIters,
                            finiteSampleBias,
                            returnPeakMap))).start()
            numSubmitted += stride
            numBusyCores += 1
    if(returnPeakMap):
        return(nPeaks, peakListTensor, peakMapTensor)
    else:
        return(nPeaks, peakListTensor)
