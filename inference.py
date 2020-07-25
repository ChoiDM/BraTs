from __future__ import print_function

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True

import os
import random
import numpy as np
from tqdm import tqdm
from glob import glob
import SimpleITK as sitk

from options import parse_option
from network import create_model
from utils.transforms import ResizeImage, decode_preds

import warnings
warnings.filterwarnings('ignore')


# Seed
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

#NOTE: main loop for training
if __name__ == "__main__":
    # Option
    opt = parse_option(print_option=False)

    # Network
    net = create_model(opt)

    # Load Patient List
    patList = glob(os.path.join(opt.data_root, 'test', '*'))

    # Inference
    for patDir in tqdm(patList[48:]):
        # Load Image Paths
        img_paths = [os.path.join(patDir, '%s_stripped.nii.gz'%img_type) for img_type in ['FLAIR', 'T1GD', 'T1', 'T2']]

        # Input Image (FLAIR, T1GD, T1, T2 order)
        imgs = [sitk.GetArrayFromImage(sitk.ReadImage(path)) for path in img_paths]
        
        # Get Shape Information
        org_size = imgs[0].shape
        meta = {'org_size' : torch.Tensor([org_size])}

        # Stack images
        imgs = [img[None, ...] for img in imgs]
        imgs = np.concatenate(imgs, axis=0)
        imgs = imgs.astype(np.float32)

        # 3D Inference
        if opt.in_dim == 3:
            MEAN = np.array(opt.mean)[:,None,None,None]
            STD = np.array(opt.std)[:,None,None,None]

            imgs = ResizeImage(imgs, (opt.in_depth, opt.in_res, opt.in_res))
            imgs = (imgs - MEAN) / STD

            # Load Data
            imgs = torch.Tensor(imgs[None,...]).float()
            if opt.use_gpu:
                imgs = imgs.cuda(non_blocking=True)

            pred = net(imgs).cpu()
        
        # 2D Inference
        elif opt.in_dim == 2:
            MEAN = np.array(opt.mean)[:,None,None]
            STD = np.array(opt.std)[:,None,None]
            
            pred = torch.zeros(opt.n_classes, imgs.shape[1], opt.in_res, opt.in_res)
            batch_size = 32

            for i in range(0, org_size[0], batch_size):
                # Load Data as Mini-Batch
                imgs_part = np.moveaxis(imgs[:,i:i+batch_size], 0, 1)
                imgs_part_resized = np.zeros((len(imgs_part), opt.in_channels, opt.in_res, opt.in_res), dtype=np.float32)

                for j, single_batch in enumerate(imgs_part):
                    imgs_part_resized[j] = (ResizeImage(single_batch, (opt.in_res, opt.in_res)) - MEAN) / STD

                imgs_part_resized = torch.Tensor(imgs_part_resized).float().contiguous()
                if opt.use_gpu:
                    imgs_part_resized = imgs_part_resized.cuda(non_blocking=True)
                with torch.no_grad():
                    pred_part = net(imgs_part_resized).cpu().transpose(0,1)

                # Stack mini-batch prediction
                pred[:, i:i+batch_size] = pred_part
            
            pred = pred[None, ...]
        
        # Prediction to Original Size and Refine Masks
        pred_decoded = decode_preds(pred, meta, refine=True)[0]

        # Numpy Array to SimpleITK 
        ce_pred, necro_pred, peri_pred = pred_decoded.data.numpy()
        ce_pred, necro_pred, peri_pred = [sitk.GetImageFromArray(array) for array in [ce_pred, necro_pred, peri_pred]]

        # Copy Header Information from Original Input Image
        FLAIR_org = sitk.ReadImage(img_paths[0])
        ce_pred.CopyInformation(FLAIR_org)
        necro_pred.CopyInformation(FLAIR_org)
        peri_pred.CopyInformation(FLAIR_org)

        # Save Predictions to Disk
        for mask_type, mask_sitk in zip(['ce', 'necro', 'peri'], [ce_pred, necro_pred, peri_pred]):
            sitk.WriteImage(mask_sitk, os.path.join(patDir, '%s_pred_mask.nii.gz'%mask_type))