import numpy
import ctypes
peakFinderPythonLib = ctypes.cdll.LoadLibrary("./RobustPeakFinder.so")
peakFinderPythonLib.peakFinder.restype = ctypes.c_int
peakFinderPythonLib.peakFinder.argtypes = [
				ctypes.c_double, ctypes.c_double,
				numpy.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                numpy.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]


def robustPeakFinderPyFunc(	indata,
							LAMBDA = 4.0,
							SNR_ACCEPT = 20.0,
							PEAK_MAX_PIX = 25):
    peakListCheeta = numpy.zeros([25000, 4])
    szx, szxy = indata.shape
    peak_cnt = peakFinderPythonLib.peakFinder(LAMBDA, SNR_ACCEPT, 
								indata, szxy, szx, 
								PEAK_MAX_PIX, peakListCheeta)
    return peakListCheeta[:peak_cnt]
