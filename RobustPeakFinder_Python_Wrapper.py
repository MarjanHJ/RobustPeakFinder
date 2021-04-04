import numpy as np
import ctypes
import os
from multiprocessing import Process, Queue, cpu_count
import time
############ if you are going to use cython, just change this part #############
dir_path = os.path.dirname(os.path.realpath(__file__))
peakFinderPythonLib = ctypes.cdll.LoadLibrary(dir_path + '/RobustPeakFinder.so')
""""RPF main calling function:
int RobustPeakFinder(float *inData, 
               unsigned char use_Mask,
               unsigned char *inMask, 
               unsigned char use_peakMask,
               unsigned char *inPeakMask,
               unsigned char minBackMeanHasAMap,
               float *minBackMeanMap, 
               unsigned char maxBackMeanHasAMap,
               float *maxBackMeanMap, 
               unsigned char returnPeakMap,
               float *peakMap,
               float *peakList,
               float singlePhotonADU,
               int MAXIMUM_NUMBER_OF_PEAKS,
               float bckSNR, 
               float pixPAPR,
               int XPIX, 
               int YPIX, 
               int PTCHSZ,    
               int PEAK_MIN_PIX, 
               int PEAK_MAX_PIX,
               int n_optIters, 
               int finiteSampleBias,
               int downSampledSize, 
               float MSSE_LAMBDA, 
               float searchSNR,
               float highPoissonTh, 
               float lowPoissonTh)
"""
peakFinderPythonLib.RobustPeakFinder.restype = ctypes.c_int
peakFinderPythonLib.RobustPeakFinder.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ctypes.c_uint8,
    np.ctypeslib.ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"),
    ctypes.c_uint8,
    np.ctypeslib.ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"),
    ctypes.c_uint8,
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),    
    ctypes.c_uint8,
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ctypes.c_uint8,
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_float, 
    ctypes.c_float,
    ctypes.c_int, 
    ctypes.c_int, 
    ctypes.c_int, 
    ctypes.c_int, 
    ctypes.c_int, 
    ctypes.c_int, 
    ctypes.c_int, 
    ctypes.c_int, 
    ctypes.c_float, 
    ctypes.c_float, 
    ctypes.c_float, 
    ctypes.c_float]
################################################################################

def robustPeakFinderPyFunc(inData, 
                           inMask = None,
                           peakMask = None,
                           minBackMeanMap = None,
                           singlePhotonADU = 1.0,
                           maxBackMeanMap = None, 
                           bckSNR = 6.0,
                           pixPAPR = 2.0,
                           PTCHSZ = 16,
                           PEAK_MAX_PIX = 25,
                           PEAK_MIN_PIX = 1,
                           MAXIMUM_NUMBER_OF_PEAKS = 1024,
                           optIters = 6,
                           finiteSampleBias = 200,
                           downSampledSize = 100,
                           MSSE_LAMBDA = 4.0,
                           searchSNR = None,
                           highPoissonTh = 0,
                           lowPoissonTh = 0,
                           returnPeakMap = False):
    """Robust Peak Finder parameters
    The Python Wrapper for the Robust Peak Finder
    Authors: Marjan Hadian Jazi LaTrobe University
             Alireza Sadri CFEL/DESY

    Inputs:
    ~~~~~~~
        inData : This is the 2D input image as a np 2d-array.
        inMask : This is the bad pixel mask.
            default: 1 for all pixels, with the size of inData
        peakMask : given a peak mask made by peakNet, it will not look for 
            peaks if peakMask is 0
            default: 1 for all pixels, with the size of inData
        minBackMeanMap : give darkSNR * STD of the values of pixels in the dark
            darkSNR = 4 to 6 gives good results.
            default: 0 with the size of inData
        singlePhotonADU : relate pixel values to variance of 
            background under high intensity Flat field
            The slope of the line passing through zero gives 
            you the single photon ADU
            NOTE: using low intensity flat field will not 
            allow you model the nonlinearities
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
        MSSE_LAMBDA: how far from its average, samples of a normal density are 
            considered inliers? default: 4.0
        searchSNR: set the threshol of searching for next candidate peak to be 
            the current model + searchSNR*standard deviation.
            default: bckSNR*0.8
        highPoissonTh : maximum acceptable results for 
            the devision \mu_B/\sigma_B^2
            default : 1.33
        lowPoissonTh : minimum acceptable results for 
            the devision \mu_B/\sigma_B^2
            default : 0.67
    Output:
    ~~~~~~~
        peakList is a np 2D-array in the style of Cheetah's output for CXI files
        Each row is for each peak and coloums are:
    ----------------------------------------------------------------------------
    Mass_Center_X, Mass_Center_Y, Mass_Total, 
                Number of pixels in a peak, Maximum value, SNR
    ----------------------------------------------------------------------------

    You can get the number of peaks by YOUROUTPUTNAME.shape[0]
    """
    inDataShape = inData.shape
    if(inMask is None):
        use_Mask = 0
        inMask = np.ones(shape = inData.shape, dtype = 'uint8')
    else:
        use_Mask = 1
        inMask = inMask.astype('uint8')
    
    if(peakMask is None):
        use_peakMask = 0
        peakMask = np.ones(shape = 1, dtype = 'uint8')
    else:
        use_peakMask = 1
        peakMask = (peakMask*inMask.copy()).astype('uint8')
    
    maxBackMeanHasAMap = 0
    if(maxBackMeanMap is None):
        maxBackMeanMap = np.finfo('float32').max \
            + np.zeros(shape = 1, dtype = 'float32')
    elif(maxBackMeanMap.size>1):
        maxBackMeanMap = maxBackMeanMap.astype('float32')
        maxBackMeanHasAMap = 1
        
    minBackMeanHasAMap = 0
    if(minBackMeanMap is None):
        minBackMeanMap = (singlePhotonADU/2.0)\
            *np.ones(shape = 1, dtype = 'float32')
    elif(minBackMeanMap.size>1):
        minBackMeanMap = minBackMeanMap.astype('float32')
        minBackMeanHasAMap = 1

    inData = inData.astype('float32')
    
    peakList = np.zeros([MAXIMUM_NUMBER_OF_PEAKS, 6], dtype='float32')    
    if(returnPeakMap):
        returnPeakMap = 1
        peakMap = np.zeros(shape = inData.shape, dtype = 'float32').flatten('F')
    else:
        returnPeakMap = 0
        peakMap = 0
    
    if(searchSNR is None):
        searchSNR = bckSNR*0.8

    peak_cnt = peakFinderPythonLib.RobustPeakFinder(inData.flatten('F'), 
                                           use_Mask,
                                           inMask.flatten('F'),
                                           use_peakMask,
                                           peakMask.flatten('F'),
                                           minBackMeanHasAMap,
                                           minBackMeanMap.flatten('F'),
                                           maxBackMeanHasAMap,
                                           maxBackMeanMap.flatten('F'),
                                           returnPeakMap,
                                           peakMap,
                                           peakList,
                                           singlePhotonADU,
                                           MAXIMUM_NUMBER_OF_PEAKS,
                                           bckSNR, 
                                           pixPAPR,
                                           inDataShape[0], 
                                           inDataShape[1], 
                                           PTCHSZ, 
                                           PEAK_MIN_PIX, 
                                           PEAK_MAX_PIX,
                                           optIters,
                                           finiteSampleBias,
                                           downSampledSize,
                                           MSSE_LAMBDA,
                                           searchSNR,
                                           highPoissonTh,
                                           lowPoissonTh)
    if(returnPeakMap):
        return((peakList[:peak_cnt], 
                peakMap.reshape(inData.shape[1], 
                inData.shape[0]).T))
    else:
         return(peakList[:peak_cnt])
    
def robustPeakFinderPyFunc_multiprocFunc(queue, 
                                         baseImgCnt, 
                                         inDataT, 
                                         inMaskT,
                                         peakMaskT,
                                         minBackMeanMap,
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
                                         downSampledSize,
                                         MSSE_LAMBDA,
                                         searchSNR,
                                         highPoissonTh,
                                         lowPoissonTh,
                                         returnPeakMap):
    for imgCnt in range(inDataT.shape[0]):
        toUnpack = robustPeakFinderPyFunc( 
            inData = inDataT[imgCnt],
            inMask = inMaskT[imgCnt],
            peakMask = peakMaskT[imgCnt],
            minBackMeanMap = minBackMeanMap[imgCnt], 
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
            downSampledSize = downSampledSize,
            MSSE_LAMBDA = MSSE_LAMBDA,
            searchSNR = searchSNR,
            highPoissonTh = highPoissonTh,
            lowPoissonTh = lowPoissonTh,
            returnPeakMap = returnPeakMap)
        if(returnPeakMap):
            peakList, peakMap = toUnpack
            queue.put(list([baseImgCnt+imgCnt, peakList, peakMap]))
        else:
            peakList = toUnpack
            queue.put(list([baseImgCnt+imgCnt, peakList]))
            #print('inQeueu ->' + str(baseImgCnt+imgCnt))

def robustPeakFinderPyFunc_multiproc(inData, 
                                     inMask = None,
                                     peakMask = None,
                                     minBackMeanMap = None, 
                                     singlePhotonADU = 1,
                                     maxBackMeanMap = None, 
                                     bckSNR = 6.0,
                                     pixPAPR = 2.0,
                                     PTCHSZ = 16,
                                     PEAK_MAX_PIX = 25,
                                     PEAK_MIN_PIX = 1,
                                     MAXIMUM_NUMBER_OF_PEAKS = 1024,
                                     optIters = 8,
                                     finiteSampleBias = 200,
                                     downSampledSize = 100,
                                     MSSE_LAMBDA = 4.0,
                                     searchSNR = None,
                                     highPoissonTh = 0,
                                     lowPoissonTh = 0,
                                     multiproc_stride = None,
                                     returnPeakMap = False):
    inDataShape = inData.shape
    if(inMask is None):
        inMask = np.ones(inDataShape, dtype='uint8')
    if(peakMask is None):
        peakMask = inMask.copy()
    if(maxBackMeanMap is None):
        maxBackMeanMap = np.ones(inDataShape, 
                                 dtype='float32')*np.finfo('float32').max
    if(minBackMeanMap is None):
        minBackMeanMap = np.zeros(inDataShape, dtype='float32')
    if(searchSNR is None):
        searchSNR = bckSNR*0.8

    f_N = inData.shape[0]
    
    peakListTensor = np.zeros((f_N, MAXIMUM_NUMBER_OF_PEAKS, 6), dtype='float32')
    if(returnPeakMap):
        peakMapTensor = np.zeros(shape = inDataShape, dtype = 'float32')
    nPeaks = np.zeros(f_N, dtype='uint32')
    
    queue = Queue()
    mycpucount = cpu_count()
    numProc = f_N
    numSubmitted = 0
    numProcessed = 0
    numBusyCores = 0
    if (mycpucount >= numProc):
        multiproc_stride = 1
    if(multiproc_stride is None):
        multiproc_stride = np.maximum(mycpucount, int(numProc/mycpucount/2))
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
            #print('numProcessed ->' + str(numProcessed))
            continue;   #its allways good to empty the queue

        if((numSubmitted<numProc) & (numBusyCores < mycpucount)):
            baseImgCnt = numSubmitted
            stride = multiproc_stride
            if(stride + numSubmitted > numProc):
                stride = numProc - numSubmitted
            inds = np.arange(baseImgCnt, baseImgCnt+stride).astype('int')
            Process(target = robustPeakFinderPyFunc_multiprocFunc, 
                    args=((queue, 
                           baseImgCnt,
                           inData[inds], 
                           inMask[inds],
                           peakMask[inds],
                           minBackMeanMap[inds],
                           singlePhotonADU,
                           maxBackMeanMap[inds],
                           bckSNR, 
                           pixPAPR,
                           PTCHSZ,
                           PEAK_MAX_PIX,
                           PEAK_MIN_PIX,
                           MAXIMUM_NUMBER_OF_PEAKS,
                           optIters,
                           finiteSampleBias,
                           downSampledSize,
                           MSSE_LAMBDA,
                           searchSNR,
                           highPoissonTh,
                           lowPoissonTh,
                           returnPeakMap))).start()
            numSubmitted += stride
            numBusyCores += 1
            #print('numSubmitted ->' + str(numSubmitted))
            
    if(returnPeakMap):
        return(nPeaks, peakListTensor, peakMapTensor)
    else:
        return(nPeaks, peakListTensor)