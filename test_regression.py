import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, RandomSampler
import numpy as np
import os, glob
import time
import pickle
import matplotlib.pyplot as plt

#----Helper Functions----#
def transform_y(y):
    return y/m0_scale

def inv_transform(y):
    return y*m0_scale

def mae_loss_wgtd(pred, true, wgt=1.):
    loss = wgt*(pred-true).abs().cuda()
    return loss.mean()

#----Dataset from H5 class----#
class H5Dataset(Dataset):
    #Initialize the Dataset object
    def __init__(self, filename, label):
        self.filename = filename
        self.label = label
        with h5py.File(filename, 'r') as f:
            self.length = len(f['all_jet'])
    #Get an image and associated quantities from the dataset
    def __getitem__(self, index):
        with h5py.File(self.filename, 'r') as f:
            img = f["all_jet"][index] 
            data = {}
            data['X_jet'] = torch.tensor(img, dtype=torch.float32)
            data['X_jet'][0] = pt_scale   * data['X_jet'][0] #Track pT
            data['X_jet'][1] = dz_scale   * data['X_jet'][1] #Track dZ sig
            data['X_jet'][2] = d0_scale   * data['X_jet'][2] #Track d0 sig
            data['X_jet'][3] = ecal_scale * data['X_jet'][3] #ECAL
            data['X_jet'][4] = hcal_scale * data['X_jet'][4] #HCAL        
            data['am']       = transform_y(np.float32(f["am"][index]))
            data['iphi']     = np.float32(f["iphi"][index])/360.
            data['ieta']     = np.float32(f["ieta"][index])/140.
            data['label']    = self.label
            # Preprocessing
            # High Value Suppressuib
            data['X_jet'][1][data['X_jet'][1] < -1] = 0  #(20 cm)
            data['X_jet'][1][data['X_jet'][1] >  1] = 0  #(20 cm)
            data['X_jet'][2][data['X_jet'][2] < -1] = 0  #(10 cm)
            data['X_jet'][2][data['X_jet'][2] >  1] = 0  #(10 cm)
            # Zero-Suppression
            data['X_jet'][0][data['X_jet'][0] < 1.e-2] = 0. #(1 GeV)
            data['X_jet'][3][data['X_jet'][3] < 1.e-2] = 0. #(0.1 GeV)
            data['X_jet'][4][data['X_jet'][4] < 1.e-2] = 0. #(0.01 GeV)
            
             #Remove some channels
            indices = [3]
            newdata = [data['X_jet'][index,:,:] for index in indices]
            data['X_jet'] = np.reshape(newdata, (1,125,125))

            
            return dict(data)
    #Get the length of the dataset (number of images in dataset)
    def __len__(self):
        return self.length



#----Constants----#
hcal_scale  = 1
ecal_scale  = 0.02
pt_scale    = 0.01
dz_scale    = 0.05
d0_scale    = 0.1
m0_scale    = 1.1

expt_name = "a_to_ele_ele_mreg_onlyECAL"

#----Parameters----#
BATCH_SIZE = 16
run_logger = False

#----Load datasets----#
test_decays = glob.glob('test_m1p0_ctau1p0_0.h5')
dset_test = ConcatDataset([H5Dataset(d, i) for i,d in enumerate(test_decays)])
n_test = ( len(dset_test) // BATCH_SIZE ) * BATCH_SIZE
test_loader    = DataLoader(dataset=dset_test, batch_size=BATCH_SIZE, num_workers=64, pin_memory=True)

#----Create Network----#
import torch_resnet_concat as networks
resblocks = 3
resnet = networks.ResNet(13, resblocks, [16, 32])
resnet.cuda()

#----Choose a trained model to use for test----#
file_name = "MODELS/a_to_ele_ele_mreg_onlyECAL/model_epoch14_a_to_ele_ele_mreg_onlyECAL_mae0.1862.pkl"
gen_mass = 1.0
model_name = file_name.split('/')[2]
print("Model name: %s"%(model_name))
checkpoint = torch.load(file_name)
resnet.load_state_dict(checkpoint['model_state_dict'])

#----Test Loop----#

loss_ = 0.
m_pred_, m_true_, mae_, mre_ = [], [], [], []
iphi_, ieta_ = [], []
ma_low = transform_y(3.6) # convert from GeV to network units
for i, data in enumerate(test_loader):
    if (i % BATCH_SIZE == 0):
        print(f"{i} of {len(test_loader)}")
    X, m0 = data['X_jet'].cuda(), data['am'].cuda()
    iphi, ieta = data['iphi'].cuda(), data['ieta'].cuda()
    logits = resnet([X, iphi, ieta])
    loss_ += mae_loss_wgtd(logits, m0).item()
    logits, m0 = inv_transform(logits), inv_transform(m0)
    mae = (logits-m0).abs()
    mre = (((logits-m0).abs())/m0)
    #Append batch metrics
    mae_.append(mae.tolist())
    mre_.append(mre.tolist())
    m_true_.append(m0.tolist())
    m_pred_.append(logits.tolist())

mae_    = np.concatenate(mae_)
mre_    = np.concatenate(mre_)
m_pred_ = np.concatenate(m_pred_)
m_true_ = np.concatenate(m_true_)
print('Test loss:%f, mae:%f, mre:%f'%(loss_/len(test_loader), np.mean(mae_), np.mean(mre_)))
score_str = '%s_mae%.4f'%(expt_name, np.mean(mae_))

#Save the results of the test
results_dict = {}
results_dict["mpred"] = m_pred_
results_dict["mtrue"] = m_true_
results_dict["name"] = model_name
results_dict["mae"] = np.mean(mae_)
results_dict["mre"] = np.mean(mre_)
results_dict["loss"] = loss_/len(test_loader)

with open('test.pickle', 'wb') as handle:
    pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Plot the test
min_mass = -0.1
max_mass = 1.2

plt.hist(np.asarray(m_pred_)[:,0],bins=26,range=[min_mass,max_mass])

#Plot vertical line for gen mass
plt.axvline(x=gen_mass, linestyle='dashed', color='black')

plt.xlabel('predicted mass (GeV)')
plt.ylabel('Events / 0.05 GeV')
plt.title('Mass regression test inference')
plt.savefig(f"test.png")

#Optional: e-mail result
#command_str = f'echo -e "Hi,\n\nAttached is the requested file.\n\nRegards,\nColin Crovella" | mailx -s "File Attachment" -a /home/cccrovella/pileup_massregression/test.png cccrovella@crimson.ua.edu'
#os.system(command_str)
