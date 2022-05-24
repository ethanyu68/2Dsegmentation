import h5py
import numpy as np
import nibabel as nib
import cv2
import os
import argparse
from utils import *
'''
This script will read NIFTI files from a folder and generate H5 file for <input> and <label>.
dir_nii: directory of NIFTI folders
Note:
1. the structure of dir_nii should follow:
    -- dir_nii:
        -- subject A
            -- xxxxxx.nii
            -- xxxxxx_label.nii
        -- subject B
            -- xxxxxx.nii
            -- xxxxxx_label.nii
        ......

2. the NIFTI file for label should end with '_label.nii' to allow program recognize it.

'''
parser = argparse.ArgumentParser(description='read nii file and save all the images in to a h5 file')
parser.add_argument("--dir_nii", type=str, default='./tr', help='the directory that stores the subjects')
parser.add_argument("--saveH5", type=str, default="tmp.h5")
parser.add_argument("--rotation", type=int, default=3, help='rotation_times of rotating the image')
parser.add_argument("--savePNG", action='store_true', help='if type --savePNG, PNG files will be saved to the NIFTI folder')
opt = parser.parse_args()

path_dir = opt.dir_nii # path of directory saving the NIFTI folders
list_nii = os.listdir(opt.dir_nii) # list of subject folders containing NIFTI files under the path_dir
num_nii = len(list_nii) # number of subject
# placeholder for images and labels and ID's
images = []
labels = []
scans_name = []

for subject in list_nii: # <subject>: the name of the subject folder
    path_subject = os.path.join(path_dir, subject) # the full path to the subject folder
    list_files = [file for file in os.listdir(path_subject) if file.endswith('nii')] 
    if 'label' in list_files[0]:
        file_label, file_orig = list_files[0], list_files[1]
    else:
        file_orig, file_label = list_files[0], list_files[1]
    # file_label: file name of label
    # file_orig: file name of orignal MRI
    path_orig = os.path.join(path_subject, file_orig)
    path_seg = os.path.join(path_subject, file_label)
    
    img_nii = nib.load(path_orig) # nii file of MRI image
    img_np = img_nii.get_fdata().__array__() # get image data and convert it to numpy array
    H, W, N = img_np.shape 
    print('shape of stack-{}: {}'.format(subject, img_np.shape))
    
    label_nii = nib.load(path_seg) # nii file of label
    label_np = label_nii.get_fdata().__array__() # get label data and convert it to numpy array
    rotation_times = opt.rotation # the images read from nii file are rotated. 
    for i in range(N):
        img = img_np[:, :, i] # take out individual slice and process
        label = label_np[:, :, i] 
        if np.sum(label) == 0:
            # if the image have no nonzero pixel, skip this one
            continue
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC) # first, resize the image to 512 x 512
        # second find the upper bound for the intensity of the image,
        #  we select the intensity that is greater than 99.9% non-zero pixels of this image.
        up_threshold = np.percentile(img[img != 0].ravel(), 99.9) 
        img[img > up_threshold] = up_threshold # set intensity greater than upper threshold to be the value of threshold
        img = img / up_threshold # normalize the image by the upper threshold
        img = np.rot90(img, k=rotation_times) # rotate the image by rotation_times to make the numpy array aligned
        if opt.savePNG:
            cv2.imwrite(os.path.join(path_subject, '{}.png'.format(i)), img * 255) # if opt.savePNG is true, save PNG file to the subject folder.
        images.append(img.tolist()) # append image to the images list
        scans_name.append(subject) # append subject name 

        
        label = cv2.resize(label, (512, 512), interpolation=cv2.INTER_NEAREST)
        label = np.rot90(label, k=rotation_times)
        labels.append(label.tolist())

        
images = np.asarray(images)
images = images[:, np.newaxis, :, :] # expand one dimension to make dataset size to be N x 1 x H x W

labels = np.asarray(labels)
labels = labels[:, np.newaxis, :, :]

h5f = h5py.File(opt.saveH5, 'w') # create a new H5 file named opt.saveH5
h5f.create_dataset(name="ID", data=np.array(scans_name, dtype='S')) 
h5f.create_dataset(name='images', data=images)
h5f.create_dataset(name='labels', data=labels)
h5f.close()
