''' -*- coding: utf-8 -*-

 @Author: kirkwood
 @Date:   2019-07-12 13:29:17
 @Email: henry.kirkwood@xfel.eu

 --------------------------------


    This is an example of how to use the wrapper for the European XFEL HDF5 files
    This works on a single AAGIPD module - the one closest to the beam. This example reads a single module of AGIPD data taken during one run in March 2018 where Lysozyme crystals were injected with a GDVN liquid jet. This example prints out the trainId, pulseId and number of peaks found for a given image. The peaks are filtered such that only peaks which are greater than one pixel in size are counted. 

    AGIPD has double size pixels at the edges of each ASIC, these are masked out in this example but should be dealt by scaling them appropriately.

    Will provide code soon to do all modules in parallel shortly

    

    Note: 
    ----------
        this example works but the mask and peak parameters need to be tuned - currently there are alot of false positives!
        



    Usage:
    --------

        The example needs to be run on the maxwell cluster, once logged into a compute node:

        # compile RobustPeakFinder
        make 
        # load python3 and required modules
        module load anaconda3

        # run example hitfinder
        python RobustPeakFinder_Python_Test_for_EuXFEL.py


    
    TODO
    ----------------------
     - tune peak finder parameters and bad pixel mask - lots of false positives
     - check chunking size of hdf5 for reading
     - only process valid pulseIds (0-64) in this case 
     - store trainIds, pulseIds and peak info
     - plot an example of the peaks found
     - apply hitfinding to all modules and combine


'''

import RobustPeakFinder_Python_Wrapper


import numpy as np
import sys
import os
import time
import glob
import itertools
import h5py
import multiprocessing as mp
import h5py
import numpy as np
import matplotlib.pyplot as plt



## define keys for hdf5 data sources (these are module dependant):
key_im = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{:d}CH0:xtdf/image/data'
key_mask = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{:d}CH0:xtdf/image/mask'
key_trainId = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{:d}CH0:xtdf/image/trainId'
key_pulseId = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{:d}CH0:xtdf/image/pulseId'

savedir = os.getcwd()



def main(proposal_dir, runNum, data_type = 'proc',):
    
    # find the desired proposal and run directory
    # glob_str =  '/gpfs/exfel/exp/SPB/*/*{}/{}/*{}'.format(proposal, data_type, runNum)
    glob_str =  proposal_dir + '{}/*{}'.format(data_type, runNum)
    runDir = glob.glob(glob_str)[0]

    print('found: {}'.format(runDir))

    # define hit finder params
    # hfParams = dict(peak_max_pix = 150,    #% maximum number of pixels in a peak
    #             peak_min_pix = 4,   #% minimum number of pixels in a peak
    #             snr_accept = 3.0,     #%...
    #             lambda_cut = 4.0)    #% 
    hfParams = dict(
                LAMBDA = 4.0,
                SNR_ACCEPT = 8.0,
                PEAK_MAX_PIX = 50)
    # agipd module number to process (0-15)
    module_num = 4

    # arg list for peak finder
    args = [runDir, module_num, hfParams]
    startT = time.time()
    # run hit finding on module for given proposal and run
    results = process_module(args)     

    endT = time.time()
    print('\nFinished in {} min'.format((endT - startT)/60.))

    
def data_sourcer(h5_filenames, module_number):
    """ function to provide agipd data one image at a time from a given set of files

    returns a generator to provide:

        trainId, pulseId, image mask, calibrated image
    """
    for f in h5_filenames:
        idx = 0
        with h5py.File(f,'r') as hf:
            # check this file has some trainIds in it:
            if isinstance(hf.get(key_trainId.format(module_number),default=-1),int): 
                print('No trainIds found in {}\n...skipping'.format(f))
                continue
            # get trainIds and pulseIds for this file and remove singelton dimension    
            trainIds = hf.get(key_trainId.format(module_number))[:].squeeze()
            pulseIds = hf.get(key_pulseId.format(module_number))[:].squeeze()
            for n in range(trainIds.shape[0]):
                calImg = hf.get(key_im.format(module_number))[n]
                calMask = hf.get(key_mask.format(module_number))[n]
                yield trainIds[n], pulseIds[n], calMask, calImg



def process_module(argsin):
    """ this function does handles the hitfinding in each module. This is called by main()
    and returns an array containing rows of trainIds, pulseIds, and found peak infos (COM etc. from robust peak finder)
    """
    runDir, module_number, hfParams = argsin
    
    fnames = get_module_files(runDir, module_number)
    
    mask = get_mask(module_number)
    
    #Mass_Center_X, Mass_Center_Y, Mass_Total, Number of pixels in a peak
    # (Nimages, Npeaks_per image, peakinfos (4)  )
    peakList_total = np.zeros((5000,4)) # holder of peaks for the whole run
    trainId_list = np.zeros(5000)
    pulseId_list = np.zeros(5000)
    pcounter = 0  # count total number of peaks
    im_counter = 0
    dSource = data_sourcer(fnames[:1], module_number)
    
    print('trainId,pulseId, Npeaks')
    # now iterate through images
    for d in dSource:
        im_counter += 1
        # unpack data from generator:
        trainId, pulseId, calMask, im = d
        if trainId<1: # skip if DAQ didn't write data
            continue
        # join my mask with the mask from the calibration pipeline together
        mask_total = (calMask==0)*mask*1.
        # find peaks
        pList = RobustPeakFinder_Python_Wrapper.robustPeakFinderPyFunc(im, mask_total, **hfParams)
        # remove peaks that are only one pixel in size ( maybe need a better mask here?)
        pList = pList[ (pList[:,-1]>1) ]
        print('{},{},{}'.format(trainId, pulseId, pList.shape[0]))

    print('processed {} images'.format(im_counter))
    return 0



def get_mask(mod_num):
    # mask the edge pixels of agipd for now - these are double size so just ignore them for now.
    # edges are True in this array, so flip it
    edges_mask = (np.load('agipd_edges_mask_stacked.npy') ==False)
    # make it a float so I can multiply with the data to apply
    return edges_mask[mod_num]
#########################################
#
#   plot result
#
#########################################
def plot_result(im,peakList):
    plt.imshow(im.T,vmin=0,vmax=10000);plt.colorbar()
    plt.plot(pList[:,1],pList[:,0],'o',markerfacecolor="None")


#########################################
#
#   DAQ handling 
#
#########################################
def get_module_files(runDir, module_number):
    """ get all the hdf5 file names associated with a given run and specific AGIPD module
        return a list of file names
    """
    if (runDir[-1] !='/'):
        runDir = runDir + '/'
    fnames = glob.glob(runDir + '*AGIPD{:02d}*.h5'.format(module_number))
    fnames.sort()
    return fnames

def chunks(l, n):
    """Yield successive n-sized chunks from l.
    This is used to break up the number of trains read 
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

##########################################
if __name__ == '__main__':

    # this directory contains data from a SPB/SFX Lysozyme experiment (there are more runs here)
    proposal_dir = '/gpfs/exfel/exp/XMPL/201750/p700000/'
    runNum = 2

    main(proposal_dir, runNum)

