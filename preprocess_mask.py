import os
import numpy as np
from glob import glob
from tqdm import tqdm
import SimpleITK as sitk

from utils.transforms import mask_binarization, fill_holes

# Description
# --- This python script generates Refined CE mask and PERI-tumoral mask
# --- using original NECRO, CE, and T2 mask.
# --- First, it generates SUM mask, which is a combined mask of original NECRO and CE mask.
# --- After, using cv2.floodfill, the missing holes of SUM mask is filled.
# --- Refined CE mask is (Filled SUM mask - original Necro mask)
# --- PERI-tumoral mask is (original T2 mask - Filled SUM mask)


# Load Patient Directory List
data_root = 'data'
patDirs = glob(os.path.join(data_root, '*', '*'))

print("Start Mask Preprocessing...")
for patDir in tqdm(patDirs):
    try:
        # Read Mask Files and Binarization
        mask_list = [os.path.join(patDir, '%s_mask.nii.gz'%mask_type) for mask_type in ['t2', 'ce', 'necro']]
        T2_sitk, CE_sitk, NECRO_sitk = [sitk.ReadImage(path) for path in mask_list]
        T2_arr, CE_arr, NECRO_arr = [sitk.GetArrayFromImage(img) for img in [T2_sitk, CE_sitk, NECRO_sitk]]
        T2_arr, CE_arr, NECRO_arr = [mask_binarization(mask) for mask in [T2_arr, CE_arr, NECRO_arr]]
        
        # Create Refined CE Mask
        SUM_arr = ((CE_arr != 0) | (NECRO_arr != 0)).astype(np.uint8)
        FILLED_SUM_arr = fill_holes(SUM_arr)
        
        CE_refined_arr = FILLED_SUM_arr.copy()
        CE_refined_arr[NECRO_arr != 0] = 0
        
        # Create Peri-tumoral Mask
        PERI_arr = T2_arr.copy()
        PERI_arr[FILLED_SUM_arr != 0] = 0
        
        # Save Generated Masks
        CE_refined_sitk, PERI_sitk = [sitk.GetImageFromArray(arr) for arr in [CE_refined_arr, PERI_arr]]
        CE_refined_sitk.CopyInformation(CE_sitk)
        PERI_sitk.CopyInformation(T2_sitk)
        
        sitk.WriteImage(CE_refined_sitk, os.path.join(patDir, 'ce_refined_mask.nii.gz'))
        sitk.WriteImage(PERI_sitk, os.path.join(patDir, 'peri_mask.nii.gz'))
    
    except Exception as e:
        print(patDir, "Error", e)