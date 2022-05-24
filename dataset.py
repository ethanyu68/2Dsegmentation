import pandas as pd
import torch.utils.data as data
import torch
import numpy as np
from skimage.transform import rotate
from skimage import transform

import scipy.ndimage as ndimage   
import matplotlib.pyplot as plt
import torch
import cv2
import h5py
from utils import *


class Dataset(data.Dataset):
    def __init__(self, path_h5, aug = True):
        '''
        path_h5: path of the h5 file
        aug: whether do augmentation
        '''
        super(Dataset, self).__init__()
        h5f = h5py.File(path_h5, 'r') # read the h5 file
        self.data = h5f['images'] # 
        self.label = h5f['labels']
        self.aug = aug
        self.num_img = self.data.shape[0]

    def augment(self, r, image, label):
        # r: a vector of uniformaly distributed random variable
        # image: 512 x 512, range 0-1
        # label: 512 x 512, values: 0,1,2,...
        # rotation
        if r[0] < 0.4:
            angle = (r[0] - 0.2) * 100 # angle range: [-20, 20]
            image = rotate(image, angle=angle, mode='wrap')
            label = rotate(label, angle=angle, mode='wrap')
        # cropping and resizing
        if r[1] < 0.3: 
            H, W = image.shape
            k1, k3 = r[2]/4, r[3]/4 # k1: [0, 0.25], k3: [0, 0.25]
            k2, k4 = 1 - r[4]/4, 1 - r[5]/4 # k2: [0.75, 1], k4:[0.75, 1]
            image = cv2.resize(image[int(k1 * H): int(k2 * H), int(k3 * W): int(k4 * W)], (512, 512),
                               interpolation=cv2.INTER_NEAREST)
            label = cv2.resize(label[int(k1 * H): int(k2 * H), int(k3 * W): int(k4 * W)], (512, 512),
                               interpolation=cv2.INTER_NEAREST)
        # resizing and padding
        if r[6] < 0.3:
            H, W = image.shape
            k1 = 1 - r[7] / 4 # k1: [0.75, 1]
            k2 = 1 - r[8] / 4 # k2: [0.75, 1]
            image_rs = cv2.resize(image, (int(k2 * W), int(k1 * H)), interpolation=cv2.INTER_NEAREST)
            label_rs = cv2.resize(label, (int(k2 * W), int(k1 * H)), interpolation=cv2.INTER_NEAREST)
            image = np.zeros([H, W])
            H_start = np.random.randint(0, np.ceil(H * (1 - k1)))
            W_start = np.random.randint(0, np.ceil(W * (1 - k2)))
            image[H_start: H_start + int(k1 * H), W_start: W_start + int(k2 * W)] = image_rs
            label[H_start: H_start + int(k1 * H), W_start: W_start + int(k2 * W)] = label_rs

        # horizonal flipping
        if r[9] < 0.3:
            image = cv2.flip(image, flipCode=1)
            label = cv2.flip(label, flipCode=1)
        # intensity scale
        if r[10] < 0.6 and r[11] < 0.6:
            miu_csf = np.mean(image[label == 1])
            miu_tissue = np.mean(image[label == 2])
            if not np.isnan(miu_tissue) and not np.isnan(miu_csf):
                miu_csf_n = miu_csf + (r[12]-0.5)/5 # the new miu_csf if between -0.1 ~ 0.1 of miu_csf
                miu_csf_n = np.min([1, miu_csf_n]) # make sure miu_csf_n is no greater than 1
                miu_tissue_n = miu_tissue + (r[13]-0.5)/5 # the new miu_csf if between -0.1 ~ 0.1 of miu_csf
                # intensity scaling
                image[image < miu_tissue] = image[image < miu_tissue] * miu_tissue_n / (miu_tissue) 
                image[(image < miu_csf) & (image > miu_tissue)] = miu_tissue_n + \
                                                                  (image[(image < miu_csf) & (
                                                                              image > miu_tissue)] - miu_tissue) \
                                                                  * (miu_csf_n - miu_tissue_n) / (miu_csf - miu_tissue)
                image[image > miu_csf] = miu_csf_n + (image[image > miu_csf] - miu_csf) * (1 - miu_csf_n) / (
                            1 - miu_csf)
        # gaussian noise ~ (0, var)
        if r[14] < 0.3: 
            image = image + np.random.normal(0, 0.005 + (r[14] - 0.15)/100, (512,512))

        # deform
        if r[15] < 0.5:
            # select four points of interest
            points_of_interest = np.array([[128, 128],
                                  [128, 384],
                                  [384, 128],
                                  [384, 384]])
            range_warp = 44
            # projection: the coordinates of points of interest after warping
            projection = np.array([[128 + np.random.randint(-range_warp, range_warp), 128 + np.random.randint(-range_warp, range_warp)],
                         [128 + np.random.randint(-range_warp, range_warp), 384+ np.random.randint(-range_warp, range_warp)],
                         [384 + np.random.randint(-range_warp, range_warp), 128+ np.random.randint(-range_warp, range_warp)],
                         [384 + np.random.randint(-range_warp, range_warp), 384+ np.random.randint(-range_warp, range_warp)]])
            tform = transform.estimate_transform('projective', points_of_interest, projection)
            image = transform.warp(image, tform.inverse, mode='edge')
            label = transform.warp(label, tform.inverse, mode='edge')
        return image, label

    def __getitem__(self, item):

        img = self.data[item, 0, :, :]
        label = self.label[item, 0, :, :]

        if self.aug:
            r = np.random.uniform(0, 1, 20) # 0~1 uniformly distributed random vector
            img, label = self.augment(r, img, label)
        img = img[np.newaxis, :, :]
        label = label[np.newaxis, :, :]
        label = np.round(label) # label might be changed after resizing. Round them to nearest integers
        return torch.from_numpy(img.copy()).float(), torch.from_numpy(label.copy()).float() # convert to float tensor. .copy() is due to issue occured in flipping


    def __len__(self):
        return self.num_img

