import torch
from torch.utils.data import Dataset

import os
import numpy as np
from glob import glob
import SimpleITK as sitk

from utils.transforms import ResizeImage, center_crop, mask_binarization, augment_imgs_and_masks

class SevBraTsDataset3D(Dataset):
    def __init__(self, data_root, opt, is_Train=True, augmentation=False):
        super(SevBraTsDataset3D, self).__init__()

        self.patientDirs = glob(os.path.join(data_root, 'train' if is_Train else 'valid', '*'))
        self.len = len(self.patientDirs)
        self.augmentation = augmentation
        
        self.in_res = opt.in_res
        self.in_depth = opt.in_depth

        self.mean = np.array(opt.mean, dtype=np.float32)[:,None,None,None]
        self.std = np.array(opt.std, dtype=np.float32)[:,None,None,None]

        self.is_Train = is_Train

    def __getitem__(self, index):
        # Patient Info
        patDir = self.patientDirs[index]
        patID = patDir.split(os.sep)[-1]

        # Load Image and Mask List
        img_paths = [os.path.join(patDir, '%s_stripped.nii.gz'%img_type) for img_type in ['FLAIR', 'T1GD', 'T1', 'T2']]
        mask_paths = [os.path.join(patDir, '%s_mask.nii.gz'%mask_type) for mask_type in ['ce_refined', 'necro', 'peri']]

        # Find 'ce.nii.gz' instead of 'ce_refined.nii.gz'
        if not os.path.isfile(mask_paths[0]):
            mask_paths[0] = mask_paths[0].replace('ce_refined', 'ce')

        # Input Image (FLAIR, T1GD, T1, T2 order)
        imgs = [sitk.GetArrayFromImage(sitk.ReadImage(path)) for path in img_paths]
        imgs = [ResizeImage(img, (self.in_depth, self.in_res, self.in_res)) for img in imgs]
        imgs = [img[None, ...] for img in imgs]

        # Ground-truth Masks (CE, NECRO, Peri order)
        masks = [sitk.GetArrayFromImage(sitk.ReadImage(path)) for path in mask_paths]
        masks_cropped = [ResizeImage(mask, (self.in_depth, self.in_res, self.in_res)) for mask in masks]
        masks_cropped = [mask_binarization(mask) for mask in masks_cropped]
        masks_cropped = [mask[None, ...] for mask in masks_cropped]

        # Stack images and Z-score Normalization
        imgs = np.concatenate(imgs, axis=0)
        imgs = imgs.astype(np.float32)

        # Remove duplicated pixels
        masks_cropped[2][masks_cropped[0]==1] = 0
        masks_cropped[2][masks_cropped[1]==1] = 0
        masks_cropped[1][masks_cropped[0]==1] = 0

        # Stack masks
        background_mask = np.ones_like(masks_cropped[0])
        background_mask[(masks_cropped[0]==1) | (masks_cropped[1]==1) | (masks_cropped[2]==1)] = 0
        masks_cropped = np.concatenate([background_mask]+masks_cropped, axis=0)
        masks_cropped = masks_cropped.astype(np.float32)

        # Augmentation
        if self.augmentation:
            pass
        
        # Z-score Normalization
        imgs = (imgs - self.mean) / self.std

        if self.is_Train:
            return imgs, masks_cropped
        
        else:
            # Stack original masks
            masks = [mask_binarization(mask) for mask in masks]
            masks = [mask[None, ...] for mask in masks]
            masks_org = np.concatenate(masks, axis=0)
            masks_org = masks_org.astype(np.float32)

            org_size = torch.Tensor(masks[0].shape[1:])
            patID = torch.Tensor([float(patID)])

            meta = {'org_size' : org_size,
                    'patientID' : patID}

            return imgs, masks_cropped, masks_org, meta
        
    def __len__(self):
        return self.len



class SevBraTsDataset2D(Dataset):
    def __init__(self, data_root, opt, is_Train=True, augmentation=False):
        super(SevBraTsDataset2D, self).__init__()

        self.data_list = glob(os.path.join(data_root, 'train' if is_Train else 'valid', '*', '2D_slice', '*.npy'))
        self.len = len(self.data_list)

        self.augmentation = augmentation
        self.rot_factor = opt.rot_factor
        self.scale_factor = opt.scale_factor
        self.flip = opt.flip
        self.trans_factor = opt.trans_factor

        self.in_res = opt.in_res

        self.mean = np.array(opt.mean, dtype=np.float32)[:,None,None]
        self.std = np.array(opt.std, dtype=np.float32)[:,None,None]

        self.is_Train = is_Train

    def __getitem__(self, index):
        # Patient Info
        slice_path = self.data_list[index]
        patID = slice_path.split(os.sep)[-1].split('_')[0]

        # Load Slice Directory
        slice_dict = np.load(slice_path, allow_pickle=True).item()

        # Input Image (FLAIR, T1GD, T1, T2 order)
        imgs = [slice_dict[img_type] for img_type in ['FLAIR', 'T1GD', 'T1', 'T2']]

        imgs = [ResizeImage(img, (self.in_res, self.in_res)) for img in imgs]
        imgs = [img[None, ...] for img in imgs]

        # Ground-truth Masks (CE, NECRO, Peri order)
        masks = [slice_dict['%s_mask'%mask_type] for mask_type in ['ce', 'necro', 'peri']]
        masks = [mask_binarization(mask) for mask in masks]
        masks = [mask[None, ...] for mask in masks]
        masks_resized = [ResizeImage(mask, (self.in_res, self.in_res)) for mask in masks]
        
        # Stack images
        imgs = np.concatenate(imgs, axis=0)
        imgs = imgs.astype(np.float32)

        # Remove duplicated pixels
        masks_resized[2][masks_resized[0]==1] = 0
        masks_resized[2][masks_resized[1]==1] = 0
        masks_resized[1][masks_resized[0]==1] = 0

        # Stack cropped masks
        background_mask = np.ones_like(masks_resized[0])
        background_mask[(masks_resized[0]==1) | (masks_resized[1]==1) | (masks_resized[2]==1)] = 0
        masks_resized = np.concatenate([background_mask]+masks_resized, axis=0)
        masks_resized = masks_resized.astype(np.float32)
        
        # Augmentation
        if self.augmentation:
            imgs, masks_resized = augment_imgs_and_masks(imgs, masks_resized, self.rot_factor, self.scale_factor, self.trans_factor, self.flip)

        # Z-Score Normalization
        imgs = (imgs - self.mean) / self.std
        
        if self.is_Train:
            return imgs, masks_resized
        
        else:
            # Stack original masks
            masks_org = np.concatenate(masks, axis=0)
            masks_org = masks_org.astype(np.float32)

            org_size = torch.Tensor(masks[0].shape[1:])
            patID = torch.Tensor([float(patID)])

            meta = {'org_size' : org_size,
                    'patientID' : patID}

            return imgs, masks_resized, masks_org, meta
        
    def __len__(self):
        return self.len