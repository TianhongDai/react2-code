import os
import torch
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import numpy as np 
import segmentation_models_pytorch as smp
from utils.preprocessing import get_training_augmentation, to_tensor, get_preprocessing

"""
this script is used to train the react2 project
"""

class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    """
    # the class for the react 2 
    CLASSES = ['c', 'r', 'bw'] 
    def __init__(
            self, images_dir, masks_dir, classes=['c', 'r', 'bw'], 
            augmentation=None, 
            preprocessing=None,
            random_scale=False,
            no_crop=False,
    ):
        self.ids = sorted(os.listdir(images_dir))
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.random_scale = random_scale
        self.preprocessing = preprocessing
        self.no_crop = no_crop
    
    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # read the mask
        suffix = self.masks_fps[i].split('.')[-1]
        prefix = self.masks_fps[:-len(suffix)-1]
        #mask = cv2.imread(self.masks_fps[i])
        mask = cv2.imread('{}.png'.format(prefix))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        # because our label is rgb states, therefore, use different type
        masks = [(mask[:,:,v] == 255) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        background = 1 - mask.sum(axis=-1, keepdims=True)
        mask = np.concatenate((background, mask), axis=-1)
        #raise NotImplementedError
        # apply preprocessing - self function!
        image, mask, paddings = self._preprocessing(image=image, mask=mask, random_scale=self.random_scale, no_crop=self.no_crop)
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if not self.no_crop:
            return image, mask
        else:
            return image, mask, paddings
    
    def _preprocessing(self, image, mask, random_scale=False, no_crop=False, fixed_size=int(72e4)):
        """
        this function is used to scale for the fixed size
        """
        # resize the image
        if random_scale: 
            scale = np.random.uniform(0.8, 1.2)
        else:
            scale = 1.0
        adjust_size = fixed_size * scale
        h, w, _ = image.shape
        ratio = w / h
        new_h = np.sqrt(adjust_size / ratio)
        new_w = adjust_size / new_h
        new_h, new_w = int(new_h), int(new_w)
        # resize
        image = cv2.resize(image, (new_w, new_h))
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        target, paddings = 32, None
        if not no_crop:
            if new_h % target != 0:
                # start to crop
                diff = new_h - (new_h // target) * target
                top_pad = diff // 2
                bottom_pad = diff - top_pad
                image = image[top_pad:-bottom_pad, :, :]
                mask = mask[top_pad:-bottom_pad, :, :]
            if new_w % target != 0:
                # start to crop
                diff = new_w - (new_w // target) * target
                left_pad = diff // 2
                right_pad = diff - left_pad
                image = image[:, left_pad:-right_pad, :]
                mask = mask[:, left_pad:-right_pad, :]
        else:
            top_pad, bottom_pad, left_pad, right_pad = 0, 0, 0, 0
            # if no crop - do the padding, and return the padding info
            if new_h % target != 0:
                # start to crop
                diff = ((new_h // target) + 1) * target - new_h
                top_pad = diff // 2
                bottom_pad = diff - top_pad
                image = np.pad(image, ((top_pad, bottom_pad), (0, 0), (0, 0)), 'reflect')
                mask = np.pad(mask, ((top_pad, bottom_pad), (0, 0), (0, 0)), 'reflect')
            if new_w % target != 0:
                # start to crop
                diff = ((new_w // target) + 1) * target - new_w
                left_pad = diff // 2
                right_pad = diff - left_pad
                image = np.pad(image, ((0, 0), (left_pad, right_pad), (0, 0)), 'reflect')
                mask = np.pad(mask, ((0, 0), (left_pad, right_pad), (0, 0)), 'reflect')
            paddings = (top_pad, bottom_pad, left_pad, right_pad)
        return image, mask, paddings
        
    def __len__(self):
        return len(self.ids)

if __name__ == '__main__':
    # set the
    im_dir = 'set_cor/train/images'
    mask_dir = 'set_cor/train/labels'
    # set encoder and encoder weights
    ENCODER = 'efficientnet-b4'
    ENCODER_WEIGHTS = 'imagenet'
    # set up the dataset
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    # create the dataset
    dataset = Dataset(im_dir, mask_dir, random_scale=True, \
                augmentation=get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
    images, labels = dataset[4]