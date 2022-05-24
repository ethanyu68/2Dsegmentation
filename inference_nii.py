import numpy as np
import torch
import cv2
from DenseNet.models_seg import Dense
import nibabel as nib
import os, glob
import argparse

from utils import *

parser = argparse.ArgumentParser(description='inference')
parser.add_argument("--folder_nii", default="test/", type=str,
                    help='path of the nii file for input')
parser.add_argument("--save_nii", default="test_results/model1/NIFTI", type=str,
                    help='path for saving the nii file')
parser.add_argument("--model", default="checkpoints/model_epoch_5.pth", type=str,
                    help='path of the neural network model')
parser.add_argument("--save_png", default="test_results/model1/PNG", type=str,
                    help='path of the folder that stores the images and outputs')
parser.add_argument("--num_output_classes", default=3, type=int, help='3 or 6, number of classes to be outputed.')
parser.add_argument("--save_format", default='nii')

opt = parser.parse_args()
model = Dense(num_classes= (opt.num_output_classes + 1), pretrain=True, block_config=((6, 6, 6),(6,6,4)))
model = model.to('cuda')
state_dict = torch.load(opt.model)['model']
if hasattr(state_dict, '_metadata'):
    del state_dict._metadata
model.load_state_dict(state_dict)

path_nii = opt.folder_nii
list_nii = os.listdir(path_nii)
for file in list_nii:
    path_file = os.path.join(path_nii, file)
    file_name = path_file.split('/')[-1]
    file_name = file_name.split('.')[0]
    path_save_nii_folder = os.path.join(opt.save_nii, file_name)
    checkdirctexist(path_save_nii_folder)
    path_save_png_folder = os.path.join(opt.save_png, file_name)
    checkdirctexist(path_save_png_folder)
    # check the path for saving the input and output

    # load nii file
    nii = nib.load(path_file)
    data = nii.get_fdata()
    # get number of images
    num_img = data.shape[2]
    seg_results = []
    print("===> The shape of the data input: {}".format(data.shape))
    for j in range(num_img):
        # the head is towards left
        img = np.rot90(data[:, :, j], k=1) # rotate 3 times counter clock-wise
        H, W = img.shape

        # Crop the image to make it multiples of 16
        if H % 16 != 0:
            h = H - H // 16 * 16
            w = W - W // 16 * 16
            img = img[h // 2:-h // 2, w // 2:-w // 2]

        # normalize the image to make it within 0 and 1

        up_threshold = np.percentile(img[img != 0].ravel(), 99.9)
        img[img > up_threshold] = up_threshold
        img = img / up_threshold
        img = normalize_numpy(img)
        data_ts = torch.tensor(img, dtype=torch.float32).cuda()

        # input to model
        with torch.no_grad():
            out = model(data_ts)

        # out: probability map in 1 x C x H x W, where C is number of classes

        out = out.cpu().numpy()
        # get prediction
        # 0: background
        # 1: CSF
        # 2: tissue
        # 3: extra axial fluid
        # 4: extra csf
        # 5: air
        pred = np.argmax(out[0], 0) # shape: H x W, values of each pixel: 0 - 5

        # if the desired number of classes is 3 (background, CSF, tissue),
        # save figure via opencv
        # save input image 
        
        cv2.imwrite(os.path.join(path_save_png_folder, '{}.png'.format(j)), unnormalize_numpy(img)[0, 0, :, :] * 255)
        cv2.imwrite(os.path.join(path_save_png_folder, '{}_seg.png'.format(j)), pred/(opt.num_output_classes)*255)
        seg_results.append(np.rot90(pred, k=-1))
    seg_results = np.array(seg_results)
    seg_results = np.rollaxis(seg_results, 0, 3)
    
    header = nii.header
    seg_result_nifti1image = nib.Nifti1Image(seg_results, np.eye(4), header)
    nib.save(seg_result_nifti1image, path_save_nii_folder + '/' + file_name + '_seg.nii')
    nib.save(nii, path_save_nii_folder + '/' + file_name + '.nii')

    print("===> The results are saved at {}".format(path_save_nii_folder))
