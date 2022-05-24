# 2Dsegmentation

## Pre-requested:
- Libraries: pytorch (1.8.0), numpy, h5py, opencv, skimage, pandas, matplotlib, scipy, nibabel


## Dataset format:
- The training and testing/validation dataset should be stored in a h5 file. In the h5file, there should be 2 datasets under the name 'images','labels'
- The shape of h5 dataset 'images' is supposed to be **N x 1 x H x W**
  - Intensity of the data should be normalized within 0 and 1. 
  - N is number of images
  - H, W are height and width
  - **H and W should be multiples of 16** (if not, downsampling and upsampling operations in the network will cause mismatch of input/output).
- The shape of h5 dataset 'label' is supposed to be **N x 1 x H x W**
  - The values of the pixels in label maps should be 0,1,..., C-1.
  - Each value represent a class.

## Model
Densely connected U-net

For processing single image:
- input: size 1x1xHxW, range 0-1
- output: 1xCxHxW, at each pixel, there are C probabilities of each class.

## Train
download the repository and run the code:
`python main_MR_seg_2Ddense.py --cuda --ID ID_of_this_model --train_data path_of_training_h5file --test_data path_of_testing_h5file`

arguments:
- batchSize: number of images stacked at each batch in training process
- train_data: path of training h5file
- test_data: path of testing h5file
- nEpochs: number of epochs to run
- lr: learning rate
- step: the number of epochs for each learning rate
- cuda: using GPU, 
- aug: if 1, do augmentation
- resume: the path of the model you want to resume to train. If not resume, place arbitrary string at this argument and it will train from the beginning.
- start-epoch: the starting epoch at training
- model: the name of the model
- ID: other information about the model

At each epoch, a model is saved at a folder under `model/{model}_{ID}/model_epoch_xx.pth`. And a figure and a csv file about the results are saved.

## Test/Inference
Install package: nibabel

To test the model, run 
`python inference_nii.py --nii <path of nii file> --model <path of the model> --save_png <path of folder to save results>`
