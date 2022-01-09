import numpy as np 
import torch
import os
import segmentation_models_pytorch as smp
from dataloader import Dataset
from torch.utils.data import DataLoader
import albumentations as albu
from datetime import datetime
import argparse
from utils.preprocessing import get_training_augmentation, to_tensor, get_preprocessing
import custom_models.deeplabv3.model

"""
this script is for the react2-seg project
"""

parser = argparse.ArgumentParser()
parser.add_argument('--train-data-path', type=str, default='set_cor/train')
parser.add_argument('--valid-data-path', type=str, default='set_cor/val')
parser.add_argument('--num-epochs', type=int, default=30, help='the number of epochs')
parser.add_argument('--save-path', type=str, default='checkpoints')
parser.add_argument('--encoder', type=str, default='resnet50', help='the name of the encoder backbone')
parser.add_argument('--arch', type=str, default='unet++', help='[unet, unet++, deeplabv3, deeplabv3+, pspnet]')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')

if __name__ == '__main__':
    # get args
    args = parser.parse_args()
    # create the dir to save ckpt
    ckpt_path = '{}_{}'.format(args.save_path, args.arch)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path, exist_ok=True)
    # start to set the training and 
    x_train_dir = '{}/images'.format(args.train_data_path)
    y_train_dir = '{}/labels'.format(args.train_data_path)
    # valid path 
    x_valid_dir = '{}/images'.format(args.valid_data_path)
    y_valid_dir = '{}/labels'.format(args.valid_data_path)
    # set encoder and encoder weights
    ENCODER = args.encoder
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multicalss segmentation
    # must use cuda for the training
    DEVICE = 'cuda'
    CLASSES = ['c', 'r', 'bw']
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
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
    # define the model
    model = model_arch(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=n_classes, #len(CLASSES), 
        activation=ACTIVATION,
        )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    # train dataset
    train_dataset = Dataset(
        x_train_dir, 
        y_train_dir, 
        augmentation=get_training_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
        random_scale=True
        )
    # valid dataset
    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        augmentation=None, 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )
    # train loader and valid loader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    # loss function
    loss = smp.utils.losses.DiceLoss()
    # metrics
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]
    # optimizer
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=args.lr),
    ])
    # create epoch runners 
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )
    # valid
    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )
    # train model for 100 epochs
    max_score = 0
    # store the training information．
    x_epoch_data = []
    train_dice_loss = []
    train_iou_score = []
    valid_dice_loss = []
    valid_iou_score = []
    print('[{}] start the training, model arch: {}'.format(datetime.now(), args.arch))
    for i in range(args.num_epochs):
        # start the training
        print('epoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        # store the information
        x_epoch_data.append(i)
        train_dice_loss.append(train_logs['dice_loss'])
        train_iou_score.append(train_logs['iou_score'])
        valid_dice_loss.append(valid_logs['dice_loss'])
        valid_iou_score.append(valid_logs['iou_score'])
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model.state_dict(), '{}/best_model.pth'.format(ckpt_path))
            print('model saved!')
        # save the model for each epoch
        if i == 10:
            optimizer.param_groups[0]['lr'] = 5e-5
            print('decrease decoder learning rate to 5e-5!')
        if i == 20:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('decrease decoder learning rate to 1e-5!')
    # save the last model
    torch.save(model.state_dict(), '{}/model_{}.pth'.format(ckpt_path, str(i).zfill(2)))