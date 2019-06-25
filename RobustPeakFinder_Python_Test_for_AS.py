import RobustPeakFinder_Python_Wrapper

import hdf5plugin
import h5py
import numpy as np
import matplotlib.pyplot as plt

PEAK_MAX_PIX = 50;
SNR_ACCEPT = 8.0;
MSSE_LAMBDA = 4.0;
HIT_NUM_PEAKS_THRESHOLD = 20;

MasterFileName = YOUR_H5_MASTER_FILE_nAME
ReadFileName = YOUR_H5_DATA_FILE_NAME

file = h5py.File(MasterFileName,'r')
#FOR AusSynch
mpbdata = file['/entry/instrument/detector/detectorSpecific/pixel_mask']
mpbdata = mpbdata[()]
mpbdata[mpbdata<1]=0;
mpbdata[mpbdata>0]=-1;
mask_data = 1+mpbdata;
file.close()

file = h5py.File(ReadFileName,'r')
datatmp = file['/entry/data/data']

for Framecnt in range(1):#datatmp.shape[0]):
    indata = datatmp[Framecnt,:,:] * mask_data
    indata = indata.astype(np.double)
    outdata = robustPeakFinderWrapper.robustPeakFinderPyFunc(indata)
    print("There is " + str(outdata.shape[0]) + " peaks in this diffraction pattern!")
    showdata = outdata[:,:2]
    
    fig = plt.figure()
    plt.imshow(indata, vmin=0, vmax= 100)
    plt.scatter(showdata[:,0],showdata[:,1])
 
    fig = plt.figure()
    plt.imshow(indata, vmin=0, vmax= 100)
    plt.scatter(showdata2[:,0],showdata2[:,1])
