
import numpy as np 
import torch
import os
import segmentation_models_pytorch as smp
from dataloader import Dataset
from torch.utils.data import DataLoader
import albumentations as albu
import cv2
from tqdm import tqdm
import argparse
from utils.preprocessing import to_tensor, get_preprocessing
from utils import binarization, boxes_detection, polygon_detection
import custom_models.deeplabv3.model

"""
this script is used to evaluate the models
"""
parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='set_cor/val')
parser.add_argument('--ckpt-path', type=str, default='checkpoints')
parser.add_argument('--encoder', type=str, default='resnet50', help='the name of the encoder backbone')
parser.add_argument('--arch', type=str, default='unet++', help='[unet, unet++, deeplabv3, deeplabv3+, pspnet]')

if __name__ == '__main__':
    args = parser.parse_args()
    # color label
    colors_label = {
        'background': [0, 0, 0], # RGB
        'c': [255, 0, 0], # RGB
        'r': [0, 255, 0], # RGB
        'bw': [0, 0, 255], # RGB
        }
    # start to set the training and 
    x_test_dir = '{}/images'.format(args.data_path)
    y_test_dir = '{}/labels'.format(args.data_path)
    # set encoder and encoder weights
    ENCODER = args.encoder
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multicalss segmentation
    DEVICE = 'cuda'
    CLASSES = ['c', 'r', 'bw']
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    # loss function
    loss = smp.utils.losses.DiceLoss()
    # metrics
    metrics = [smp.utils.metrics.IoU(threshold=0.5),]
    # set up the UNet plus plus
    if args.arch == 'unet++':
        model_arch = smp.UnetPlusPlus
    elif args.arch == 'unet':
        model_arch = smp.Unet
    elif args.arch == 'deeplabv3':
        model_arch = custom_models.deeplabv3.model.DeepLabV3
    elif args.arch == 'deeplabv3+':
        model_arch = custom_models.deeplabv3.model.DeepLabV3Plus
    elif args.arch == 'pspnet':
        model_arch = smp.PSPNet
    elif args.arch == 'fpn':
        model_arch = smp.FPN
    elif args.arch == 'manet':
        model_arch = smp.MAnet
    elif args.arch == 'linknet':
        model_arch = smp.Linknet
    else:
        raise NotImplementedError
    # set the model
    model = model_arch(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=n_classes, #len(CLASSES), 
        activation=ACTIVATION,
        )
    model.load_state_dict(torch.load('{}'.format(args.ckpt_path), map_location='cpu'))
    model.cuda()
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    # test dataset
    test_dataset = Dataset(
        x_test_dir, 
        y_test_dir, 
        augmentation=None, 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    # test epoch
    test_epoch = smp.utils.train.ValidEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )
    # evaluate the model
    logs = test_epoch.run(test_loader)