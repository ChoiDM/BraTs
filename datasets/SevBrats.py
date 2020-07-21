from torch.utils.data import Dataset

import os
import numpy as np
from glob import glob
import SimpleITK as sitk

from utils.transforms import ResizeImage, center_crop

class SevBraTsDataset3D(Dataset):
    def __init__(self, data_root, opt, is_Train=True, augmentation=False):
        super(SevBraTsDataset3D, self).__init__()

        self.patientDirs = glob(os.path.join(data_root, 'train' if is_Train else 'valid', '*'))
        self.len = len(self.patientDirs)
        self.augmentation = augmentation
        
        self.mean = np.array(opt.mean, dtype=np.float32)[:,None,None,None]
        self.std = np.array(opt.std, dtype=np.float32)[:,None,None,None]

    def __getitem__(self, index):
        # Patient Info
        patDir = self.patientDirs[index]
        patID = patDir.split(os.sep)[-1]

        # Load Image and Mask List
        img_paths = [os.path.join(patDir, '%s_stripped.nii.gz'%img_type) for img_type in ['FLAIR', 'T1GD', 'T1', 'T2']]
        mask_paths = [os.path.join(patDir, '%s_mask.nii.gz'%mask_type) for mask_type in ['necro', 'ce_refined', 'peri']]

        # Input Image (FLAIR, T1GD, T1, T2 order)
        imgs = [sitk.GetArrayFromImage(sitk.ReadImage(path)) for path in img_paths]
        imgs = [center_crop(img, 180, 180) for img in imgs]
        imgs = [ResizeImage(img, (180, 180, 144)) for img in imgs]
        imgs = [img[None, ...] for img in imgs]

        # Ground-truth Masks (NECRO, CE, Peri order)
        masks = [sitk.GetArrayFromImage(sitk.ReadImage(path)) for path in mask_paths]
        masks = [center_crop(mask, 180, 180) for mask in masks]
        masks = [ResizeImage(mask, (180, 180, 144)) for mask in masks]
        masks = [mask[None, ...] for mask in masks]

        # Augmentation
        if self.augmentation:
            pass
        
        # Stack images and Z-score Normalization
        imgs = (np.concatenate(imgs, axis=0) - self.mean) / self.std
        imgs = imgs.astype(np.float32)

        # Stack masks
        background_mask = np.ones_like(masks[0])
        background_mask[(masks[0]==1) | (masks[1]==1) | (masks[2]==1)] = 0
        masks = np.concatenate([background_mask]+masks, axis=0)
        masks = masks.astype(np.float32)

        return imgs, masks
        
    def __len__(self):
        return self.len



class SevBraTsDataset2D(Dataset):
    def __init__(self, data_root, is_Train=True, augmentation=False):
        super(SevBraTsDataset2D, self).__init__()
        self.data_list = glob(os.path.join(data_root, 'train' if is_Train else 'valid', '*', '*.npy'))
        self.len = len(self.data_list)
        self.augmentation = augmentation
        self.is_Train = is_Train

    def __getitem__(self, index):
        slice_dict = np.load(self.data_list[index], allow_pickle=True).item()

        # Input Image
        flair = slice_dict['FLAIR']
        t1ce = slice_dict['T1CE']
        t1 = slice_dict['T1']
        t2 = slice_dict['T2']

        # Mask
        ce_mask = slice_dict['ce_mask']
        necro_mask = slice_dict['necro_mask']
        peri_mask = slice_dict['peri_mask']

        # Augmentation
        if self.augmentation:
            pass

        # Add axis
        flair, t1ce, t1, t2, ce_mask, necro_mask, peri_mask = [arr[None, ...] for arr in [flair, t1ce, t1, t2, ce_mask, necro_mask, peri_mask]]
        
        bg_mask = np.ones_like(ce_mask)
        bg_mask[(ce_mask==1) | (necro_mask==1) | (peri_mask==1)] = 0

        # Ground-truth Mask
        mask_gt = np.concatenate([bg_mask, necro_mask, peri_mask, ce_mask], axis=0)
        
        img = (np.concatenate([flair, t1, t1ce, t2], axis = 0))
        mask_gt = mask_gt.astype(np.float32)

        return img, mask_gt
        
    def __len__(self):
        return self.len