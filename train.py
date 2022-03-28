import numpy as np
import os
import torch
import fnmatch
from  skimage.io import imread
from RobustGaussianFittingLibrary import fitValue
import LossFunc
import networks
import datos
#import matplotlib.pyplot as plt

learning_rate      = 1e-4
device             = 'cpu'
DATOS_order        = 'ascend'
Fit2LeastSeenRate  = 2
fit2What           = None#'worst' #'worst'#'leastSeen'
fit2FixedK         = 0.95
infer_size         = 1
n_k                = 10
sampleSize         = 4
n_epochs           = 5
n_kSweeps          = 32
indUpdate_showProg = True
OutliersThreshold  = None
plotMSSEs          = True
sortAt             = None#'eachkSweep'#eachkSweep' #'eachkUpdate'
plotRelativeOrders = True
includeCloseOutliers = False
near_far_comboSize = 4
dataDriven_LR_and_epochs = False
dataDriven_LR_near = 5e-3
dataDriven_LR_far = 1e-2
dataDriven_nEpochs_near = 4
dataDriven_nEpochs_far = 12
dataDriven_sampleSize_near = 200
dataDriven_sampleSize_far = 20
lr_schedule = None
#lr_schedule        = 1e-6+np.zeros(n_kSweeps)
#lr_schedule_filler = np.array([1e-3, 1e-3, 5e-4, 1e-4, 5e-5, 
#                               1e-5, 5e-6, 1e-6])
#lr_schedule[:lr_schedule_filler.shape[0]] = lr_schedule_filler

usePretrained      = False
n_test             = 500
n_train            = 100000 + n_test

pretrainedFileName = './model.model'
scratchFolder = './'
logDIR = scratchFolder + 'cellNet_logs/'
dSetDIR = scratchFolder + 'data/'
ingput_DIR = dSetDIR + 'images/'
output_DIR = dSetDIR + 'masks/'

str2log  = 'learning_rate      --> '+ str(learning_rate     ) + '\n'
str2log += 'device             --> '+ str(device            ) + '\n'
str2log += 'DATOS_order        --> '+ str(DATOS_order       ) + '\n'
str2log += 'n_test             --> '+ str(n_test            ) + '\n'
str2log += 'n_train            --> '+ str(n_train           ) + '\n'
str2log += 'infer_size         --> '+ str(infer_size        ) + '\n'
str2log += 'n_k                --> '+ str(n_k               ) + '\n'
str2log += 'sampleSize         --> '+ str(sampleSize        ) + '\n'
str2log += 'n_epochs           --> '+ str(n_epochs          ) + '\n'
str2log += 'n_kSweeps          --> '+ str(n_kSweeps         ) + '\n'
str2log += 'indUpdate_showProg --> '+ str(indUpdate_showProg) + '\n'
str2log += 'OutliersThreshold  --> '+ str(OutliersThreshold ) + '\n'
str2log += 'usePretrained      --> '+ str(usePretrained     ) + '\n'
str2log += 'plotMSSEs          --> '+ str(plotMSSEs         ) + '\n'
str2log += 'sortAt             --> '+ str(sortAt            ) + '\n'
str2log += 'plotRelativeOrders --> '+ str(plotRelativeOrders) + '\n'
str2log += 'includeCloseOutliers --> '+ str(includeCloseOutliers) + '\n'
str2log += '-----------------------------------------\n'
print(str2log)

def n_correctFunc(preds, labels):
    predictions = 0*preds.copy()
    for ptCbt in range(preds.shape[0]):
        mP = fitValue(preds[ptCbt,0].flatten(),
                      downSampledSize = 1000,
                      MSSE_LAMBDA=2.5)
        threshold_low = mP[0] - 2.5 * mP[1]
        threshold_high = mP[0] + 2.5 * mP[1]
        predictions[ptCbt, preds[ptCbt] < threshold_low] = 1
        predictions[ptCbt, preds[ptCbt] > threshold_high] = 1
    inds = np.where( ( (labels == 0) & (predictions == 0) ) \
                   | ( (labels == 1) & (predictions == 1) ) )[0]
    return(inds.shape[0])

class dataMaker:
    def __init__(self, ingput_DIR, output_DIR):
        self.ingput_DIR = ingput_DIR
        self.output_DIR = output_DIR
        
        netInputs_flist = self.get_flist_from_dir(self.ingput_DIR)
        netOutputs_flist = self.get_flist_from_dir(self.output_DIR,
                                            fileNameTemplate = '*.png')

        netInputs_flist_cpy = netInputs_flist.copy()
        for idx in range(len(netInputs_flist)):
            netInputs_flist_cpy[idx] = netInputs_flist[idx].split('.')[0]
        netOutputs_flist_cpy = netOutputs_flist.copy()
        for idx in range(len(netOutputs_flist)):
            netOutputs_flist_cpy[idx] = netOutputs_flist[idx].split('.')[0]

        in_set = set(netInputs_flist_cpy)
        out_set = set(netOutputs_flist_cpy)
        flist_cpy = list(in_set.intersection(out_set))

        self.netInputs_flist = flist_cpy.copy()
        for idx in range(len(flist_cpy)):
            self.netInputs_flist[idx] = flist_cpy[idx] + '.tif'
        self.netOutputs_flist = flist_cpy.copy()
        for idx in range(len(flist_cpy)):
            self.netOutputs_flist[idx] = flist_cpy[idx] + '.png'

        self.n_pts = len(flist_cpy)
        self.netInputs_all = self.load_images(\
            self.ingput_DIR, self.netInputs_flist, np.arange(self.n_pts))
        self.netOutputs_all = self.load_images(\
            self.output_DIR, self.netOutputs_flist, np.arange(self.n_pts))
        self.netOutputs_all = self.netOutputs_all[..., 0]
        
    def get_flist_from_dir(self, theDIR, fileNameTemplate = '*.tif'):
        flist = fnmatch.filter(os.listdir(theDIR), fileNameTemplate)
        flist.sort()
        return(flist)
    
    def load_images(self, theDIR, flist, sampleIndices):
        img = imread(theDIR + flist[sampleIndices[0]])
        img = img[:512, :640].copy()
        img = np.array([img])
        img_all = np.zeros((sampleIndices.shape[0], ) + img.shape,
                       dtype = img.dtype)
        for cnt in range(sampleIndices.shape[0]):
            img = imread(theDIR + flist[sampleIndices[0]])
            img_all[cnt, 0, :, :] = img[:512, :640].copy()

        return img_all.astype('float')
        
    def __call__(self, sampleIndices):
        try:
            _ = sampleIndices.shape[0]
        except:
            sampleIndices = np.array([sampleIndices])
        netInput00 = self.netInputs_all[sampleIndices]
        netOutput00 = self.netOutputs_all[sampleIndices]
        netInput = torch.tensor(netInput00).float()
        netOutput = torch.tensor(netOutput00).float()
        return(netInput, netOutput)

if __name__ == '__main__':

    dataMakerFuncObj = dataMaker(ingput_DIR, output_DIR)
    n_pts = dataMakerFuncObj.n_pts
    print('Datamaker initialized with ', n_pts, ' number of data points')

    model = networks.U_Net(img_ch=1).float().to(device)
    os.system('python3 gpu_stat.py')
    if(usePretrained):
        print('Using pretrained')
        model.load_state_dict(torch.load(pretrainedFileName), strict=False)
        usePretrained = False
        
    print('Network loaded into GPU')
    criterion = LossFunc.diffraLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, 
    #                                    model.parameters()), 
    #                             lr = learning_rate, weight_decay=1e-6)

    subX, subY = dataMakerFuncObj(0)
    labels = model(torch.from_numpy(subX.numpy().astype('float')).float().to(device))
    if(device == 'cuda'):
        criterion(subY, labels)
    else:
        criterion(subY.numpy(), labels.detach().numpy())
    
    classes = np.ones(n_pts, dtype = 'int')
    DATOS_sampler = datos.datos(\
        model,
        criterion, 
        optimizer,
        device, 
        dataMakerFuncObj, 
        infer_size, 
        classes = classes, 
        n_k = n_k, 
        sampleSize = sampleSize, 
        order=DATOS_order, 
        n_epochs = n_epochs, 
        n_kSweeps = n_kSweeps,
        n_correctFunc = n_correctFunc,
        indicesUpdate_showProgress = indUpdate_showProg,
        logDIR = logDIR,
        numTestDataPoints = n_test,
        Fit2LeastSeenRate = Fit2LeastSeenRate,
        fit2What = fit2What,
        OutliersThreshold = OutliersThreshold,
        plotMSSEs = plotMSSEs,
        sortAt = sortAt,
        plotRelativeOrders = plotRelativeOrders,
        fit2FixedK = fit2FixedK,
        logTitle = str2log,
        includeCloseOutliers = includeCloseOutliers,
        lr_schedule = lr_schedule,
        dataDriven_LR_and_epochs = dataDriven_LR_and_epochs,
        dataDriven_LR_near = dataDriven_LR_near,
        dataDriven_LR_far = dataDriven_LR_far,
        dataDriven_nEpochs_near = dataDriven_nEpochs_near,
        dataDriven_nEpochs_far = dataDriven_nEpochs_far,
        dataDriven_sampleSize_near = dataDriven_sampleSize_near,
        dataDriven_sampleSize_far = dataDriven_sampleSize_far,
        near_far_comboSize = near_far_comboSize)
        
    keepTraining = True
    while(keepTraining):
        DATOS_state = DATOS_sampler.trainStep()
        if(DATOS_state == 'continue'):
            continue
        else:
            print('DATOS just gave up! ')
            print('Check your log directory ->' \
                + DATOS_sampler.logDIR, flush=True)
            exit()
