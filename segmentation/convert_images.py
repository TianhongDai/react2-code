import numpy as np 
import cv2
import os
from tqdm import tqdm
import argparse
"""
this script is to re-ognize the image
"""

parser = argparse.ArgumentParser()
parser.add_argument('--im-path', type=str, default='react2-code/react_data/images')
parser.add_argument('--save-path', type=str, default='react2-code/react_data/correct_images')

if __name__ == '__main__':
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    im_list = os.listdir(args.im_path)
    lst = []
    for name in tqdm(im_list):
        suffix = name.split('.')[-1]
        prefix = name[:-len(suffix)-1]
        tf_im = cv2.imread('{}/{}'.format(args.im_path, name))
        cv2.imwrite('{}/{}.png'.format(args.save_path, prefix), tf_im)