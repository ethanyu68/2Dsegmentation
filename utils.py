import numpy as np
import math, os
import pdb
import torch
import shutil
def checkdirctexist(dirct):
	if not os.path.exists(dirct):
		os.makedirs(dirct)


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW)
    """
    #assert input.dim() == 4

    # expand the input tensor to Nx1xDxHxW before scattering
    input = input.unsqueeze(1)
    # create result tensor shape (NxCxDxHxW)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)


def PSNR_self(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = (pred -gt)
    pred_np = pred.cpu().images.numpy()
    gt_np = pred.cpu().images.numpy()
    rmse = math.sqrt(np.mean(imdff.cpu().images.numpy() ** 2))
    if rmse == 0:
        return 100
    return 20.0 * math.log10(1/rmse)


def adjust_learning_rate(epoch, opt):
    lr = opt.lr * (opt.lr_reduce ** (epoch // opt.step))
    return lr


def save_checkpoint(model, epoch, optimizer, opt, save_path):
    checkdirctexist(save_path)
    model_out_path = os.path.join(save_path, "model_epoch_{}.pth".format(epoch))
    state = {"epoch": epoch, "model": model.module.state_dict(), "optimizer":optimizer.state_dict()}
    # check path status
    if not os.path.exists("model/"):
        os.makedirs("model/")
    # save model_files
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    # copy python file to checkpoints folder


def normalize_tensor(data):
    N, _, H, W = data.shape
    img = torch.cat([data, data, data], 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1])
    std = torch.tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1])
    img = (img - mean) / std
    return img


def normalize_numpy(data):
    #N, _, H, W = data.shape
    img = np.concatenate([data, data, data], 1)
    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1])
    std = np.array([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1])
    img = (img - mean) / std
    return img


def unnormalize_tensor(data):
    mean = torch.tensor([0.485, 0.456, 0.406]).view([1, 3, 1, 1])
    std = torch.tensor([0.229, 0.224, 0.225]).view([1, 3, 1, 1])
    orig = data * std + mean
    return orig


def unnormalize_numpy(data):
    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1])
    std = np.array([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1])
    orig = data * std + mean
    return orig


def convert_to_cbf(map, num):
    RGB =[[245, 121, 58], [169, 90, 161], [133, 192, 249], [15, 32, 128]]
    new_map = np.zeros([map.shape[0], map.shape[1], 3])
    for c in range(num):
        new_map[map==c+1] = RGB[c]
    return new_map.astype(np.uint8)

