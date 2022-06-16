import numpy as np 
import cv2
import os
import csv
import json
from tqdm import tqdm
import shutil
import argparse

"""
this script is used to generate the dataset for training
"""

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='../images', help='the data path')
parser.add_argument('--save-path', type=str, default='react_data', help='the dir to save the data')


if __name__ == '__main__':
    # parse the arguments
    args = parser.parse_args()
    # generate the dataset
    ground_truth_path = '{}/ground_truth/seg_gtruth_readout_region.csv'.format(args.data_path)
    im_lists = os.listdir('{}/images'.format(args.data_path))
    # saved path
    saved_data_path = '{}/images'.format(args.save_path)
    saved_label_path = '{}/labels'.format(args.save_path)
    ignore_lists = ['103581_D547FBF6-A129-4BE6-A532-8B30BC73D0AE.jpeg', '103873_B76A9A44-C5B5-4D69-B9F9-C0C1FE2947C8.jpeg']
    if os.path.exists(saved_data_path):
        exist_data = os.listdir(saved_data_path)
    else:
        exist_data = []
    ignore_lists = ignore_lists + exist_data
    # create the folder
    if not os.path.exists(saved_data_path):
        os.makedirs(saved_data_path, exist_ok=True)
    if not os.path.exists(saved_label_path):
        os.makedirs(saved_label_path, exist_ok=True)   
    # check csv, and save the information into the label_info
    label_info = {}
    with open('{}'.format(ground_truth_path)) as f:
        gt_info = csv.DictReader(f)
        for info_ in gt_info:
            if info_['filename'] not in label_info.keys():
                label_info[info_['filename']] = {}
            # save the region's position
            region_label = json.loads(info_['region_attributes'])
            pos_label = json.loads(info_['region_shape_attributes'])
            label_info[info_['filename']][region_label['label']] = pos_label 
    # start to generate the dataset 
    for im_name in tqdm(label_info.keys()):
        if im_name in ignore_lists:
            continue
        if im_name not in im_lists:
            continue
        im = cv2.imread('{}/images/{}'.format(args.data_path, im_name))
        # get the name
        suffix = im_name.split('.')[-1]
        im_name_prefix = im_name[:-len(suffix)-1]
        h, w, _ = im.shape 
        # we have 4 classes
        label_ = np.zeros((h, w, 3), dtype=np.uint8) 
        all_x_pts = label_info[im_name]['c']['all_points_x']
        all_y_pts = label_info[im_name]['c']['all_points_y']
        points = [None] * len(all_x_pts)
        for i in range(len(all_x_pts)):
            x = int(all_x_pts[i])
            y = int(all_y_pts[i])
            points[i] = (x, y)
        contours = np.array(points)
        for x in range(w):
            for y in range(h):
                if cv2.pointPolygonTest(contours, (x, y), False) > 0:
                    label_[y, x, 0] = 255
                    label_[y, x, 1] = 0
                    label_[y, x, 2] = 0
        # get the position of r
        all_x_pts = label_info[im_name]['r']['all_points_x']
        all_y_pts = label_info[im_name]['r']['all_points_y']
        points = [None] * len(all_x_pts)
        for i in range(len(all_x_pts)):
            x = int(all_x_pts[i])
            y = int(all_y_pts[i])
            points[i] = (x, y)
        contours = np.array(points)
        for x in range(w):
            for y in range(h):
                if cv2.pointPolygonTest(contours, (x, y), False) > 0:
                    label_[y, x, 0] = 0
                    label_[y, x, 1] = 255
                    label_[y, x, 2] = 0
        # get the position of btw
        all_x_pts = label_info[im_name]['bw']['all_points_x']
        all_y_pts = label_info[im_name]['bw']['all_points_y']
        points = [None] * len(all_x_pts)
        for i in range(len(all_x_pts)):
            x = int(all_x_pts[i])
            y = int(all_y_pts[i])
            points[i] = (x, y)
        contours = np.array(points)
        for x in range(w):
            for y in range(h):
                if cv2.pointPolygonTest(contours, (x, y), False) > 0:
                    label_[y, x, 0] = 0
                    label_[y, x, 1] = 0
                    label_[y, x, 2] = 255
        cv2.imwrite('{}/{}.png'.format(saved_label_path, im_name_prefix), label_)
        shutil.copy2('{}/images/{}'.format(args.data_path, im_name), '{}/{}'.format(saved_data_path, im_name))