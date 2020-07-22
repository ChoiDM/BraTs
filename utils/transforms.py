import cv2
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from random import randrange


def ResizeSitkImage(sitk_file, new_shape):
    new_spacing = [org_spacing*org_size/new_size for org_spacing, org_size, new_size
                   in zip(sitk_file.GetSpacing(), sitk_file.GetSize(), new_shape)]

    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(sitk_file.GetDirection())
    resample.SetOutputOrigin(sitk_file.GetOrigin())
    resample.SetOutputSpacing(new_spacing)

    resample.SetSize(new_shape)

    return resample.Execute(sitk_file)
    
def ResizeImage(img, new_shape):
    if type(img) is np.ndarray:
        img = sitk.GetImageFromArray(img)
        img_resized = ResizeSitkImage(img, new_shape)
        return sitk.GetArrayFromImage(img_resized)
    
    else:
        return ResizeSitkImage(img, new_shape)

def center_crop(img_array, x_size, y_size):
    if np.ndim(img_array) == 4:
        _, z, y, x = img_array.shape

        x_start = (x//2) - (x_size//2)
        y_start = (y//2) - (y_size//2)

        return img_array[:, :,
                        y_start : y_start + y_size,
                        x_start : x_start + x_size]
    
    elif np.ndim(img_array) == 3:
        z, y, x = img_array.shape

        x_start = (x//2) - (x_size//2)
        y_start = (y//2) - (y_size//2)
        
        img_crop = img_array[:,
                        y_start : y_start + y_size,
                        x_start : x_start + x_size]

        return img_crop

def random_rotation(img, mask, min_angle = -15, max_angle = 15, axes=(1,2)):
    angle = randrange(min_angle, max_angle)

    img_rot = ndimage.rotate(img, angle, axes=axes, reshape=False, mode='reflect').copy()
    mask_rot = ndimage.rotate(mask, angle, axes=axes, reshape=False, mode='reflect').copy()
    return img_rot, mask_rot

def fill_holes(image):
    # Reference : https://stackoverflow.com/questions/50450654/filling-in-circles-in-opencv
    
    n_slices, h, w = image.shape
    image_filled = np.zeros_like(image, dtype=np.uint8)
    
    for d in range(n_slices):
        # Threshold
        th, im_th = cv2.threshold(image[d]*255, 127, 255, cv2.THRESH_BINARY)

        # Copy the thresholded image
        im_floodfill = im_th.copy()

        # Mask used to flood filling.
        # NOTE: the size needs to be 2 pixels bigger on each side than the input image
        h, w = im_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0,0), 255)

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground
        im_out = im_th | im_floodfill_inv
        
        image_filled[d] = im_out.copy()
        
    image_filled = (image_filled / 255.).astype(np.uint8)
    return image_filled

def mask_binarization(mask_array):
    threshold = np.max(mask_array) / 2
    mask_binarized = (mask_array > threshold).astype(np.uint8)
    
    return mask_binarized