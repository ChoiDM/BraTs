import os
import numpy as np
from glob import glob
from tqdm import tqdm
import SimpleITK as sitk

from utils.transforms import mask_binarization, fill_holes


# Load Patient Directory List
data_root = 'data'
patDirs = glob(os.path.join(data_root, '*', '*'))

print("Start 2D Data Parsing...")
for patDir in tqdm(patDirs):
    try:
        # Patient Info
        patID = patDir.split(os.sep)[-1]
        subset = patDir.split(os.sep)[-2]

        # Load Image and Mask List
        img_paths = [os.path.join(patDir, '%s_stripped.nii.gz'%img_type) for img_type in ['FLAIR', 'T1GD', 'T1', 'T2']]
        mask_paths = [os.path.join(patDir, '%s_mask.nii.gz'%mask_type) for mask_type in ['necro', 'ce_refined', 'peri']]

        # Find 'ce.nii.gz' instead of 'ce_refined.nii.gz'
        if not os.path.isfile(mask_paths[1]):
            mask_paths[1] = mask_paths[1].replace('ce_refined', 'ce')

        # Input Image (FLAIR, T1GD, T1, T2 order)
        FLAIR, T1GD, T1, T2 = [sitk.GetArrayFromImage(sitk.ReadImage(path)) for path in img_paths]

        # Ground-truth Masks (NECRO, CE, Peri order)
        masks = [sitk.GetArrayFromImage(sitk.ReadImage(path)) for path in mask_paths]
        NECRO, CE, PERI = [mask_binarization(mask) for mask in masks]

        # Save slice dictionary
        for d in range(len(FLAIR)):
            slice_dict = {'FLAIR' : FLAIR[d],
                        'T1GD' : T1GD[d],
                        'T1' : T1[d],
                        'T2' : T2[d],
                        'necro_mask' : NECRO[d],
                        'ce_mask' : CE[d],
                        'peri_mask' : PERI[d]}
            
            save_path = os.path.join(patDir, '2d_slice', '%s_%03d.npy' % (patID, d))
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
                
            np.save(save_path, slice_dict)
    
    except Exception as e:
        print(patDir, "Error", e)
