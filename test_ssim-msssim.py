from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import torch
import torchvision.transforms as transforms
from PIL import Image
import pathlib

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=256,
                    help='Batch size to use')
parser.add_argument('--num-workers', type=int, default=8,
                    help='Number of processes to use for data loading')
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--path', type=str, nargs=2,
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def calculte_ssim_msssim(XData, YData, batch_size=256, device='cpu', num_workers=8):
    if batch_size > len(XData):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(XData)

    if batch_size > len(YData):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(YData)

    Xpath = pathlib.Path(XData)
    XData = sorted([file for ext in IMAGE_EXTENSIONS
                    for file in Xpath.glob('*.{}'.format(ext))])
    Ypath = pathlib.Path(YData)
    YData = sorted([file for ext in IMAGE_EXTENSIONS
                    for file in Ypath.glob('*.{}'.format(ext))])

    Xdataset = ImagePathDataset(XData, transforms=transforms.Compose([
        transforms.ToTensor()

    ]))

    Ydataset = ImagePathDataset(YData, transforms=transforms.Compose([
        transforms.ToTensor()

    ]))
    Xdataloader = torch.utils.data.DataLoader(Xdataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=num_workers)
    Ydataloader = torch.utils.data.DataLoader(Ydataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=num_workers)


    ssim_vals = []
    msssim_vals = []
    start_idx = 0
    for Xbatch, Ybatch in zip(tqdm(Xdataloader), tqdm(Ydataloader)):
        Xbatch = Xbatch.to(device)
        Ybatch = Ybatch.to(device)
        ssim_val, msssim_val = get_MSSSIM_SSIM(Xbatch, Ybatch)

        ssim_vals.append(ssim_val)
        msssim_vals.append(msssim_val)
    return ssim_vals, msssim_vals


def get_MSSSIM_SSIM(X, Y):
    # X: (N,3,H,W) a batch of non-negative RGB images (0~255)
    # Y: (N,3,H,W)
    win_size = 3
    # calculate ssim & ms-ssim for each image
    ssim_val = ssim(X, Y, data_range=255, size_average=False, win_size=win_size)  # return (N,)
    ms_ssim_val = ms_ssim(X, Y, data_range=255, size_average=False, win_size=win_size)  # (N,)

    # set 'size_average=True' to get a scalar value as loss. see tests/tests_loss.py for more details
    ssim_loss = 1 - ssim(X, Y, data_range=255, size_average=True, win_size=win_size)  # return a scalar
    ms_ssim_loss = 1 - ms_ssim(X, Y, data_range=255, size_average=True, win_size=win_size)

    # reuse the gaussian kernel with SSIM & MS_SSIM.
    ssim_module = SSIM(data_range=255, size_average=True, channel=3, win_size=win_size)
    ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3, win_size=win_size)

    ssim_loss = 1 - ssim_module(X, Y)
    ms_ssim_loss = 1 - ms_ssim_module(X, Y)

    return ssim_val, ms_ssim_val


import numpy as np
import urllib
import time
from skimage.metrics import structural_similarity
import os
import sys
import torch
import tensorflow as tf


def main():
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    ssim, msssim = calculte_ssim_msssim(args.path[0],
                                        args.path[1],
                                        args.batch_size,
                                        args.device,
                                        args.num_workers)
    ssim = np.mean(ssim)
    msssim = np.mean(msssim)
    print('SSIM: ', ssim)
    print('MS-SSIM: ', msssim)


if __name__ == '__main__':
    main()