import os

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torch.utils.data import DataLoader
from dataset import Dataset
import matplotlib.pyplot as plt
from DenseNet.models_seg import Dense
#import torchgeometry as tgm
import torchvision.models as models
import numpy as np
from losses import DiceLoss
import argparse
from utils import *
import shutil
import json

parser = argparse.ArgumentParser()
parser.add_argument("--block_config", default=((6, 6, 6),(6,6,4)), type=tuple)
parser.add_argument("--batchsize", default=8, type=int, help='batch size in training')
parser.add_argument("--batchsize_val", default=32, type=int, help='batch size in validation')
parser.add_argument("--initial_lr", default=0.0001, type=float, help='initial learning rate')
parser.add_argument("--height", default=512, type=int, help='height of the image')
parser.add_argument("--width", default=512, type=int, help='width of the image')
parser.add_argument("--ep_start", default=1, type=int, help='starting epoch')
parser.add_argument("--ep_end", default=100, type=int, help='ending epoch')
parser.add_argument("--lr_reduction", default=0.2, type=float, help='the learning rate will be reduced to <lr_reduction> of current rate at every <step size>')
parser.add_argument("--step_size", default=20, type=int, help='learning rate will be reduced at every <step_size> epoch')
parser.add_argument("--model_ID",default='', help='the name of the folder where models are saved')
parser.add_argument("--path_tr",default='tmp.h5', help='the training dataset')
parser.add_argument("--path_val",default='tmp.h5', help='the validation dataset')
parser.add_argument("--path_model",default='DenseNet/models_seg.py', help='the model ID to be saved')
parser.add_argument("--path_main",default='train.py', help='the path of main file to be saved')
parser.add_argument("--path_dataset",default='dataset.py', help='the path of datset file to be saved')
parser.add_argument("--pretrain_ImageNet", default=True)
parser.add_argument("--path_resume",default="...", help='the model ID to be saved')
parser.add_argument("--results_table", default='results_table.csv')
##
parser.add_argument("--notation", default='Description:.')
opt= parser.parse_args()
def main():
    print(opt)
    global classes
    # check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # set dice loss
    print("===> Setting Loss function")
    dice_loss = DiceLoss(sigmoid_normalization=False)
    # path for saving checkpoints
    path_ckp = os.path.join('checkpoints/', opt.model_ID)
    checkdirctexist(path_ckp)
    shutil.copy(opt.path_model, path_ckp) # save model file to checkpoint folder
    shutil.copy(opt.path_main, path_ckp) # save main file to checkpoint folder
    shutil.copy(opt.path_dataset, path_ckp) # save dataset file to checkpoint folder
    # save training parameters
    with open(path_ckp + '/opt.txt', 'w') as f:
        json.dump(opt.__dict__, f, indent=2) # save parameters
    path_logs = os.path.join('runs', opt.model_ID) 
    # if tensorboard logs exist, delete them and create new one
    if os.path.exists(path_logs): 
        shutil.rmtree(path_logs)
    # create new logs
    writer = SummaryWriter(path_logs)
    # read CSV table that contains names of classes
    results = pd.read_csv(opt.results_table) 
    classes = results.keys()[1:]
    print("===> The output classes:{}".format(classes)) 
    # set up model
    print("===> Setting Model")
    model = Dense(num_classes=len(classes), pretrain=opt.pretrain_ImageNet, block_config=opt.block_config)
    # if opt.path_resume exist, load this model and train it from the saved epoch
    if os.path.exists(opt.path_resume):
        state = torch.load(opt.path_resume)['model']
        model.load_state_dict(state)
        opt.ep_start = torch.load(opt.path_resume)['epoch']
    # if multiple GPUs are being used, the line below will be useful
    model = torch.nn.DataParallel(model).to(device)
    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.initial_lr, weight_decay=1e-6)
    print("===> Setting Dataset")
    train_set = Dataset(path_h5=opt.path_tr, aug=True) # augmentation is set True
    val_set = Dataset(path_h5=opt.path_val, aug=False) # augmentation is set False
    train_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchsize, shuffle=True)
    val_data_loader = DataLoader(dataset=val_set, batch_size=opt.batchsize_val, shuffle=False)
    # make a new table for saving results
    results = {'epoch':[]}
    for key in classes:
        results[key] = []
    # training
    # training will start at epoch <opt.ep_start> and end at epoch <opt.ep_end>
    for epoch in range(opt.ep_start, opt.ep_end):
        train(train_data_loader, optimizer, model, epoch, dice_loss, opt, writer)
        # at every 5 epochs, validate the model and saved the validation results and model parameters
        if epoch % 5 == 0:
            # obtain dice loss of each class (channel in output maps)
            per_channel_loss = validate(val_data_loader, model, epoch, dice_loss, opt, writer)
            # add <epoch> to result table
            results['epoch'].append(epoch)
            for c, cls in enumerate(classes):
                # add dice score to the table
                results[cls].append(1 - float('{:.4f}'.format(per_channel_loss[c])))
            # save result table as CSV file named opt.results_table
            df = pd.DataFrame(results)
            df.to_csv(path_ckp+opt.results_table)
            # save model parameters
            save_checkpoint(model, epoch, optimizer, opt, path_ckp)

def train(training_data_loader, optimizer, model, epoch, criterion, opt, writer):
    lr = opt.initial_lr * (opt.lr_reduction ** (epoch // opt.step_size))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        data, label= batch
        # normalize the input image according to requirements of pre-trained DenseNet
        data = normalize_tensor(data) 
        data = data.cuda()
        label = label.cuda() 
        # input data to model
        # out: size B x C x H x W
        out = model(data)
        # calculate dice loss
        loss, per_channel_loss = criterion(out, expand_as_one_hot(label[:, 0, :, :].long(),
                                                                  C= len(classes)))  # out[1] = 1, if there is objec
        # print training loss at every 50 iteration and 
        # write results to tensorboard
        if iteration % 10 == 0:
            print("===> Epoch[{}]({}), lr: {}, ce:{:.6f}".format(epoch, iteration, lr, loss))
            writer.add_images("Train/Image", unnormalize_tensor(data.cpu()))
            writer.add_images("Train/Out", torch.argmax(out, 1).view(-1, 1, opt.height, opt.width) / len(classes))
            writer.add_images("Train/Label", label / len(classes))
            writer.flush()
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # at end of each epoch, add losses to tensorboard
    writer.add_scalar("Training_loss/average", loss.cpu(), epoch)
    for c, cls in enumerate(classes):
        writer.add_scalar("Training_loss/{}".format(cls), per_channel_loss[c].cpu(), epoch)
    writer.flush()


def validate(val_data_loader, model, epoch, criterion, opt, writer):
    model.eval()
    loss_all = 0
    per_channel_loss_all = np.zeros(len(classes))
    iter_to_display = np.random.randint(len(val_data_loader))
    for iteration, batch in enumerate(val_data_loader, 1):
        with torch.no_grad():
            data, label = batch
            data = normalize_tensor(data)
            out = model(data.to('cuda'))
        loss, per_channel_loss = criterion(out.cpu(), expand_as_one_hot(label[:, 0, :, :].long(), C= len(classes)))
        loss_all += loss
        for c, channel_loss in enumerate(per_channel_loss):
            per_channel_loss_all[c] += channel_loss
        # display images in tensorboard
        if iteration == iter_to_display:
            r = np.random.randint(0, data.shape[0] - 4)
            images_to_display = unnormalize_tensor(data[r:4 + r])[:, 0:1, :, :]
            labels_to_display = label[r:4 + r] / len(classes)
            outputs_to_display = torch.argmax(out[r:4 + r], 1).view(-1, 1, opt.height, opt.width).cpu()/len(classes)
            display_tensors = torch.cat([images_to_display, labels_to_display, outputs_to_display], 0)
            writer.add_images("Val/Image_batch", display_tensors)
    # compute average loss across the whole validataion dataset
    loss_all = loss_all / iteration
    for c in range(len(classes)):
        per_channel_loss_all[c] = per_channel_loss_all[c]/iteration
    # write validation loss to tensorboard
    writer.add_scalar("Validation_loss/average", loss_all, epoch)
    for c, cls in enumerate(classes):
        writer.add_scalar("Validation_loss/{}".format(cls), per_channel_loss[c].cpu(), epoch)
    writer.flush()

    return per_channel_loss.cpu().numpy()




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
