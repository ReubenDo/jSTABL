import numpy as np
import os
import nibabel as nib
import SimpleITK as sitk

def find_zeros(img_array):
    if len(img_array.shape) == 4:
        img_array = np.amax(img_array, axis=3)
    assert len(img_array.shape) == 3
    x_dim, y_dim, z_dim = tuple(img_array.shape)
    x_zeros, y_zeros, z_zeros = np.where(img_array == 0.)
    # x-plans that are not uniformly equal to zeros
    
    try:
        x_to_keep, = np.where(np.bincount(x_zeros) < y_dim * z_dim)
        x_min = min(x_to_keep)
        x_max = max(x_to_keep) + 1
    except Exception :
        x_min = 0
        x_max = x_dim
    try:
        y_to_keep, = np.where(np.bincount(y_zeros) < x_dim * z_dim)
        y_min = min(y_to_keep)
        y_max = max(y_to_keep) + 1
    except Exception :
        y_min = 0
        y_max = y_dim
    try :
        z_to_keep, = np.where(np.bincount(z_zeros) < x_dim * y_dim)
        z_min = min(z_to_keep)
        z_max = max(z_to_keep) + 1
    except:
        z_min = 0
        z_max = z_dim
    return x_min, x_max, y_min, y_max, z_min, z_max


def crop(list_img):
    img_crop = nib.load(list_img[0])
    affine = img_crop.affine
    img_crop_data = img_crop.get_fdata()
    x_min, x_max, y_min, y_max, z_min, z_max = find_zeros(img_crop_data)

    x_max = img_crop_data.shape[0] - x_max
    y_max = img_crop_data.shape[1] - y_max
    z_max = img_crop_data.shape[2] - z_max
    bounds_parameters = [x_min, x_max, y_min, y_max, z_min, z_max]
    low = bounds_parameters[::2]
    high = bounds_parameters[1::2]
    low = [int(k) for k in low]
    high = [int(k) for k in high]
 
    for path_mod in list_img:
        image = sitk.ReadImage(path_mod)
        image = sitk.Crop(image, low, high)
        sitk.WriteImage(image, path_mod)
