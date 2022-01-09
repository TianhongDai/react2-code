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

def crop_paddings(im, paddings):
    # crop
    top_pad, bottom_pad, left_pad, right_pad = paddings
    # remove pad
    if top_pad * bottom_pad != 0:
        im = im[:, top_pad:-bottom_pad, :]
    else:
        if top_pad == 0 and bottom_pad != 0:
            im = im[:, :-bottom_pad, :]
        elif top_pad !=0 and bottom_pad == 0:
            im = im[:, top_pad:]
    # remove pad
    if left_pad * right_pad != 0:
        im = im[:, :, left_pad:-right_pad]
    else:
        if left_pad ==0 and right_pad != 0:
            im = im[:, :, :-right_pad]
        elif left_pad !=0 and right_pad == 0:
            im = im[:, :, left_pad:]
    return im

# resize the image corrdinates
def resize_image_coordinates(input_coordinates, input_shape, resized_shape):
    rx = input_shape[0] / resized_shape[0]
    ry = input_shape[1] / resized_shape[1]
    return np.stack((np.round(input_coordinates[:, 0] / ry),
                      np.round(input_coordinates[:, 1] / rx)), axis=1).astype(np.int32)

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='../react2-code/react_data')
parser.add_argument('--ckpt-path', type=str, default='checkpoints')
parser.add_argument('--encoder', type=str, default='resnet50', help='the name of the encoder backbone')
parser.add_argument('--arch', type=str, default='unet++', help='[unet, unet++, deeplabv3, deeplabv3+, pspnet]')
parser.add_argument('--save-path', type=str, default='results')

if __name__ == '__main__':
    args = parser.parse_args()
    # color label
    colors_label = {
        'background': [0, 0, 0], # RGB
        'c': [255, 0, 0], # RGB
        'r': [0, 255, 0], # RGB
        'bw': [0, 0, 255], # RGB
        }
    # save path for the seg and box
    seg_save_path = '{}_{}/seg'.format(args.save_path, args.arch)
    box_save_path = '{}_{}/box'.format(args.save_path, args.arch)
    # create the dir to save the results
    if not os.path.exists(seg_save_path):
        os.makedirs(seg_save_path, exist_ok=True)
    if not os.path.exists(box_save_path):
        os.makedirs(box_save_path, exist_ok=True)
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
        no_crop=True,
    )
    # draw the plots
    bad_examples = []
    for i in tqdm(range(len(test_dataset))):
        # get the prefix
        im_path = test_dataset.ids[i]
        suffix = im_path.split('.')[-1]
        prefix = im_path[:-len(suffix)-1]
        im, _, paddings = test_dataset[i]
        # read the original image
        ori_im = cv2.imread('{}/{}'.format(x_test_dir, im_path))
        ori_im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
        x_tensor = torch.from_numpy(im).to(DEVICE).unsqueeze(0)
        # get the probability mask
        with torch.no_grad():
            pr_mask = model.predict(x_tensor)
        pr_mask = pr_mask.squeeze().cpu().numpy()
        pr_mask = crop_paddings(pr_mask, paddings)
        # resize the pr mask to the original size
        color_index = np.argmax(pr_mask, axis=0).astype('uint8')
        # crop the paddings
        colors = list(colors_label.values())
        canvas = np.zeros((pr_mask.shape[1], pr_mask.shape[2], 3)).astype('uint8')
        for idx, color in enumerate(colors):
            canvas[color_index == idx] = color
        canvas = cv2.resize(canvas, (ori_im.shape[1], ori_im.shape[0]), interpolation=cv2.INTER_NEAREST)
        seg_im = ori_im // 2 + canvas // 2
        cv2.imwrite('{}/{}.png'.format(seg_save_path, prefix), seg_im[:,:,::-1])
        # try to label the regions - 
        probs = np.transpose(pr_mask, (1, 2, 0))
        binary_maps = np.zeros_like(probs, np.uint8)
        binary_maps = np.delete(binary_maps, 0, 2)
        # region 1
        probs_ = probs[:, :, 1]
        mask_ = binarization.thresholding(probs_, 0.75)
        mask_ = binarization.cleaning_binary(mask_, kernel_size=3)
        cboxes = polygon_detection.find_polygonal_regions(mask_, min_area=0., n_max_polygons=1)
        bin_map = np.zeros_like(binary_maps)
        binary_maps[:, :, 0] = cv2.fillPoly(bin_map, cboxes, (255, 0, 0))[:, :, 0]
        # region 2
        probs_ = probs[:, :, 2]
        mask_ = binarization.thresholding(probs_, 0.25)
        mask_ = binarization.cleaning_binary(mask_, kernel_size=3)
        rboxes = polygon_detection.find_polygonal_regions(mask_, min_area=0., n_max_polygons=1)
        bin_map = np.zeros_like(binary_maps)
        binary_maps[:, :, 1] = cv2.fillPoly(bin_map, cboxes, (0, 255, 0))[:, :, 1]
        # region 3
        probs_ = probs[:, :, 3]
        mask_ = binarization.thresholding(probs_, 0.75)
        mask_ = binarization.cleaning_binary(mask_, kernel_size=3)
        bwboxes = polygon_detection.find_polygonal_regions(mask_, min_area=0., n_max_polygons=1)
        bin_map = np.zeros_like(binary_maps)
        binary_maps[:, :, 2] = cv2.fillPoly(bin_map, cboxes, (0, 0, 255))[:, :, 2]
        # plot the polygon
        if cboxes and rboxes and bwboxes:
            # resize box
            original_shape = (ori_im.shape[0], ori_im.shape[1]) 
            cboxes_resized = [resize_image_coordinates(box, probs.shape[:2], original_shape) for box in cboxes]
            rboxes_resized = [resize_image_coordinates(box, probs.shape[:2], original_shape) for box in rboxes]
            bwboxes_resized = [resize_image_coordinates(box, probs.shape[:2], original_shape) for box in bwboxes]
            for box in cboxes_resized:
                cv2.polylines(ori_im, [box[:, None, :]], True, (255, 0, 0), thickness=5)
            for box in rboxes_resized:
                cv2.polylines(ori_im, [box[:, None, :]], True, (0, 255, 0), thickness=5)
            for box in bwboxes_resized:
                cv2.polylines(ori_im, [box[:, None, :]], True, (0, 0, 255), thickness=5)
        else:
            bad_examples.append(im_path)
        # write the images
        cv2.imwrite('{}/{}.png'.format(box_save_path, prefix), ori_im[:,:,::-1])
    print('bad examples: {}'.format(bad_examples))
    print('number of bad examples: {}'.format(len(bad_examples)))