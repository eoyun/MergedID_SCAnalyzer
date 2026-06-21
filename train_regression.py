import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, RandomSampler
import numpy as np
import os, glob
import time
import matplotlib.pyplot as plt

#----Argument parsing----#
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-e', '--epochs',     default=10,    type=int, help='Number of training epochs.')
parser.add_argument('-l', '--lr_init',    default=1.e-4, type=float, help='Initial learning rate.')
parser.add_argument('-b', '--resblocks',  default=3,     type=int, help='Number of residual blocks.')
parser.add_argument('-a', '--load_epoch', default=0,     type=int, help='Which epoch to start training from')
parser.add_argument('-n', '--new_lr',     default=0,     type=float, help='New learning rate when loading epoch.')
parser.add_argument('-f', '--lr_factor',  default=0.2,   type=float, help='Learning rate factor')
parser.add_argument('-p', '--patience',   default=2,     type=float, help='Learning schedule patience')
args = parser.parse_args()

lr_init = args.lr_init
new_lr = args.new_lr
resblocks = args.resblocks
epochs = args.epochs
load_epoch = args.load_epoch
lr_factor = args.lr_factor
patience = args.patience
#os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda)


#----Helper Functions----#
def transform_y(y):
    return y/m0_scale

def inv_transform(y):
    return y*m0_scale

def logger(s):
    global f, run_logger
    print(s)
    if run_logger:
        f.write('%s\n'%str(s))

def mae_loss_wgtd(pred, true, wgt=1.):
    loss = wgt*(pred-true).abs().cuda()
    return loss.mean()

def huber(pred, true, delta):
    if (true-pred).abs().cuda() < delta:
        loss = 0.5*((true-pred)**2).cuda()
    else:
        loss = delta*(pred-true).abs().cuda() - 0.5*(delta**2)
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
            # High Value Suppression
            data['X_jet'][1][data['X_jet'][1] < -1] = 0  #(20 cm)
            data['X_jet'][1][data['X_jet'][1] >  1] = 0  #(20 cm)
            data['X_jet'][2][data['X_jet'][2] < -1] = 0  #(10 cm)
            data['X_jet'][2][data['X_jet'][2] >  1] = 0  #(10 cm)
            # Zero-Suppression
            #data['X_jet'][0][data['X_jet'][0] < 1.e-2] = 0. #(1 GeV)
            data['X_jet'][0][data['X_jet'][0] < 0.5] = 0.
            data['X_jet'][3][data['X_jet'][3] < 1.e-2] = 0. #(0.1 GeV)
            data['X_jet'][4][data['X_jet'][4] < 1.e-2] = 0. #(0.01 GeV)
            
            #Remove some channels if you want to train models with only certain layers (eg only pT and ECAL)
            #indices = [0,3]
            #newdata = [data['X_jet'][index,:,:] for index in indices]
            #data['X_jet'] = np.reshape(newdata, (2,125,125))

            
            return dict(data)
    #Get the length of the dataset (number of images in dataset)
    def __len__(self):
        return self.length
        





#----Scaling Constants----#
# This keeps the various channels' pixels at roughly the same order-of-magnitude
hcal_scale  = 1
ecal_scale  = 0.02
pt_scale    = 0.01

dz_scale    = 0.05
d0_scale    = 0.1
m0_scale    = 1.2

expt_name = "a_to_ele_ele_mreg_allchannels"

#----Parameters----#
BATCH_SIZE = 256
run_logger = True

#----Load datasets----#
train_decays = glob.glob('/bighome/cccrovella/massregression_m0p01To1p2_pT25To160_pileup/IMG/train/*.h5')
val_decays = glob.glob('/bighome/cccrovella/massregression_m0p01To1p2_pT25To160_pileup/IMG/val/*.h5')

dset_train = ConcatDataset([H5Dataset(d, i) for i,d in enumerate(train_decays)])
dset_val = ConcatDataset([H5Dataset(d, i) for i,d in enumerate(val_decays)])

n_train = ( len(dset_train) // BATCH_SIZE ) * BATCH_SIZE
n_val = ( len(dset_val) // BATCH_SIZE ) * BATCH_SIZE

train_sampler = RandomSampler(dset_train, replacement=True, num_samples=n_train)
train_loader  = DataLoader(dataset=dset_train, batch_size=BATCH_SIZE, num_workers=64, pin_memory=True, sampler=train_sampler)

val_sampler   = RandomSampler(dset_val, replacement=True, num_samples=n_val)
val_loader    = DataLoader(dataset=dset_val, batch_size=BATCH_SIZE, num_workers=64, pin_memory=True, sampler=val_sampler)


#----Create Network----#
import torch_resnet_concat as networks
resnet = networks.ResNet(13, resblocks, [16, 32])
resnet.cuda()
optimizer = optim.Adam(resnet.parameters(), lr=lr_init)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=patience) 


#----Set up logger----#
if run_logger:
    if not os.path.isdir('LOGS'):
        os.makedirs('LOGS')
    f = open('LOGS/%s.log'%(expt_name), 'w')
    #for d in ['MODELS', 'METRICS','PLOTS']:
    for d in ['MODELS', 'PLOTS']:
        if not os.path.isdir('%s/%s'%(d, expt_name)):
            os.makedirs('%s/%s'%(d, expt_name))

#----Training Loop----#

logger('Number of training samples: %d'%(n_train))
logger('Number of validation samples: %d'%(n_val))

print_step = 100
mae_best = 1.
logger(">> Training <<<<<<<<")

#----Load a different model for transfer learning----#

#file_name = "MODELS/a_to_ele_ele_mreg_onlyECAL/model_epoch43_a_to_ele_ele_mreg_onlyECAL_mae0.6781.pkl"
#model_name = file_name.split('/')[2]
#print("Loaded Model name: %s"%(model_name))
#checkpoint = torch.load(file_name)
#resnet.load_state_dict(checkpoint['model_state_dict'])
#load_epoch = 14

#------Training Loop------#
for e in range(epochs):

    epoch = e+1+load_epoch
    epoch_wgt = 0.
    n_trained = 0
    logger('>> Epoch %d <<<<<<<<'%(epoch))

    resnet.train()
    now = time.time()
    
    #Do training
    for i, data in enumerate(train_loader):  #For each image in train dataset
        X, m0 = data['X_jet'].cuda(), data['am'].cuda()  #Load the image (X) and true mass of pseudoscalr (m0)
        iphi, ieta = data['iphi'].cuda(), data['ieta'].cuda()  #Load the iphi, ieta of supercluster
        optimizer.zero_grad()
        logits = resnet([X, iphi, ieta])   #Get the output of the model from this image
        loss = mae_loss_wgtd(logits, m0)   #Calculate the loss
        #loss = huber(logits, m0, 1)
        loss.backward()                    #Back propagate the loss to update the weights
        optimizer.step()
        epoch_wgt += len(m0)
        n_trained += 1
        if i % print_step == 0:  #Print out status every {print_step} images
            logits, m0 = inv_transform(logits), inv_transform(m0) 
            mae =  (logits-m0).abs().mean()
            mre = (((logits-m0).abs())/m0).mean()
            logger('%d: (%d/%d) m_pred: %s...'%(epoch, i, len(train_loader), str(np.squeeze(logits.tolist()[:5]))))
            logger('%d: (%d/%d) m_true: %s...'%(epoch, i, len(train_loader), str(np.squeeze(m0.tolist()[:5]))))
            logger('%d: (%d/%d) Train loss:%f, mae:%f, mre:%f'%(epoch, i, len(train_loader), loss.item(), mae.item(), mre.item() ))  
    now = time.time() - now
    logits, m0 = inv_transform(logits), inv_transform(m0)
    mae = (logits-m0).abs().mean()
    mre = ((logits-m0).abs()/m0).mean()
    #Print out training time and training loss for this epoch
    logger('%d: Train time:%.2fs in %d steps for N:%d, wgt: %.f'%(epoch, now, len(train_loader), n_trained, epoch_wgt))
    logger('%d: Train loss:%f, mae:%f, mre:%f'%(epoch, loss.item(), mae.item(), mre.item() ))


    #Do validation
    loss_ = 0.
    m_pred_, m_true_, mae_, mre_ = [], [], [], []
    iphi_, ieta_ = [], []
    ma_low = transform_y(3.6) # convert from GeV to network units
    for i, data in enumerate(val_loader):
        #if (i > 1):
        #    break
        X, m0 = data['X_jet'].cuda(), data['am'].cuda()
        iphi, ieta = data['iphi'].cuda(), data['ieta'].cuda()
        logits = resnet([X, iphi, ieta])
        loss_ += mae_loss_wgtd(logits, m0).item()
        logits, m0 = inv_transform(logits), inv_transform(m0)
        mae = (logits-m0).abs()
        mre = (((logits-m0).abs())/m0)
        if i % 100 == 0:
            logger('Validation (%d/%d): Train loss:%f, mae:%f, mre:%f'%(i, len(val_loader), loss_/(i+1), mae.mean().item(), mre.mean().item() ))
        #Store batch metrics
        mae_.append(mae.tolist())
        mre_.append(mre.tolist())
        m_true_.append(m0.tolist())
        m_pred_.append(logits.tolist())

    
    mae_    = np.concatenate(mae_)
    mre_    = np.concatenate(mre_)
    m_pred_ = np.concatenate(m_pred_)
    m_true_ = np.concatenate(m_true_)
    
    logger('%d: Val loss:%f, mae:%f, mre:%f'%(epoch, loss_/len(val_loader), np.mean(mae_), np.mean(mre_)))
    score_str = 'epoch%d_%s_mae%.4f'%(epoch, expt_name, np.mean(mae_))
    lr_scheduler.step(loss_/len(val_loader))


    #Save the model weights for this epoch
    filename  = 'MODELS/%s/model_%s.pkl'%(expt_name, score_str)
    loss_value = loss_/len(val_loader)
    model_dict = {'model_state_dict': resnet.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'epoch' : epoch, 'loss': loss_value}
    torch.save(model_dict, filename)

    #Validation plot
    #Plot predicted mass vs true mass
    min_mass = -0.1
    max_mass = 1.2

    plt.hist2d(np.asarray(m_true_)[:,0],np.asarray(m_pred_)[:,0],bins=(50,50),range=[[min_mass,max_mass],[min_mass,max_mass]],cmap=plt.cm.jet)

    #Plot diagonal line y = x
    x = np.linspace(min_mass, max_mass, 100)
    plt.plot(x, x, linestyle='dashed', color='black')
    
    plt.xlabel('true mass (GeV)')
    plt.ylabel('predicted mass (GeV)')
    plt.title('Mass regression validation inference')
    plt.savefig(f"PLOTS/{expt_name}/{score_str}.png")
   
    #Optional: e-mail final validation plot
    #if (e==epochs-1):
    #if (True):
    #    command_str = f'echo -e "Hi,\n\nAttached is the requested file.\n\nRegards,\nColin Crovella" | mailx -s "File Attachment" -a /home/cccrovella/pileup_massregression/PLOTS/{expt_name}/{score_str}.png cccrovella@crimson.ua.edu'
    #    os.system(command_str)
