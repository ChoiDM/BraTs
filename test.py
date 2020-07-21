#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.optim as optim
from torch.nn import DataParallel
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils.losses import DiceLoss, lovasz_softmax
from utils.metrics import DiceCoef
from network.cascaded_unet import CascadedUnet, SubUnet
from network.modified_unet import Modified3DUNet
from utils.data_loader import MICCAIBraTsDataset

import os
import numpy as np
import nibabel as nib
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR


# In[2]:


# Critical Setting
pkl_path = 'epoch_0080_lr_0.00005_loss_0.1887_dice_0.8131.pkl'


# In[3]:


# GPU Setting
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[4]:


# Train Setting
batch_size = 2

train_dir = 'MICCAI_BraTS_2019_Data_Training'


# In[5]:


# Data Loader
valid_dataset = MICCAIBraTsDataset(train_dir, is_Train=False, augmentation = augmentation)
valid_data_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False, num_workers = 0)


# In[6]:


# Model
model = Modified3DUNet(in_channels = 4, n_classes = 4, base_n_filter=16)
model = DataParallel(model)
model = model.to(device)


# Training
if __name__ == "__main__":
    # -------------------------------- Train Dataset --------------------------------
    running_dice = 0
    n_samples = 0
    model.eval()

    for step, (img, mask) in enumerate(valid_data_loader):
        # Load Data
        img, mask = Variable(img).to(device), Variable(mask).to(device)
        img, mask = img.float(), mask.float()
        # Predict
        pred  = model(img)

        dice_score = DiceCoef()(pred, mask)


            # Stack Results
            n_batch_samples = len(img)
            n_samples += n_batch_samples
            running_loss += loss.data * n_batch_samples
            running_dice += dice_score.data * n_batch_samples

            if (step==0) or (step+1) % 10 == 0:
                print('     > Step [%3d/%3d] Loss %.4f - Dice Coef %.4f' % (step+1, len(train_data_loader), running_loss/n_samples, running_dice/n_samples))
        
        train_loss = running_loss / n_samples
        train_dice = running_dice / n_samples
        
        
        # -------------------------------- Valid Dataset --------------------------------
        running_loss = 0
        running_dice = 0
        n_samples = 0
        model.eval()

        for step, (img, mask) in enumerate(valid_data_loader):
            # Load Data
            img, mask = Variable(img).to(device), Variable(mask).to(device)
            img, mask = img.float(), mask.float()

            # Predict
            with torch.no_grad():
                pred  = model(img)

            if loss_func == 'dice':
                loss = criterion(pred, mask)

            dice_score = DiceCoef()(pred, mask)


            # Stack Results
            n_batch_samples = len(img)
            n_samples += n_batch_samples
            running_loss += loss.data * n_batch_samples
            running_dice += dice_score.data * n_batch_samples
            print('     > Step [%3d/%3d] Loss %.4f - Dice Coef %.4f' % (step+1, len(valid_data_loader), running_loss/n_samples, running_dice/n_samples))

        valid_loss = running_loss / n_samples
        valid_dice = running_dice / n_samples
        
        print('     ==> Result : Train Loss %.4f - Train Dice Coef %.4f - Valid Loss %.4f - Valid Dice Coef %.4f'
                    % (train_loss, train_dice, valid_loss, valid_dice))
        
        # Save Results
        if not os.path.exists(os.path.join(save_dir, 'pkl')):
            os.makedirs(os.path.join(save_dir, 'pkl'))

        if (not save_only_best_score) or (valid_dice > best_score):
            best_score = valid_dice
            torch.save(model.state_dict(), os.path.join(save_dir, 'pkl', 'epoch_%04d_lr_%.5f_loss_%.4f_dice_%.4f.pkl' % ((epoch+1), lr, valid_loss, valid_dice)))
        
        lr=lr_update(epoch, lr_warmup_epoch, optimizer, scheduler)

