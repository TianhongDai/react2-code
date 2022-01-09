import numpy as np 
import argparse
import torch
from vit_models import ViT
from dataset_loader import ReactDataset
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from datetime import datetime
import os
from tqdm import tqdm

"""
this script is used to train the network using ViT 

please notice, class 1 is for IGG postive
"""

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='./dataset')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--train-steps', type=int, default=20000)
parser.add_argument("--warmup-steps", type=int, default=500, help='learning rate warm up steps')
parser.add_argument('--ckpt-path', type=str, default='saved_ckpt')
parser.add_argument('--cuda', action='store_true', help='if use cuda')

def cal_metric_multi_class(conf_matrix, c=1):
    TP = conf_matrix.diag()[c]
    idx = torch.ones(2).bool()
    idx[1] = 0
    # all non-class samples classified as non-class
    TN = conf_matrix[torch.nonzero(idx, as_tuple=False)[:, None], torch.nonzero(idx, as_tuple=False)].sum()
    # all non-class samples classified as class
    FP = conf_matrix[idx, c].sum()
    # all class samples not classified as class
    FN = conf_matrix[c, idx].sum()
    return TP.item(), TN.item(), FP.item(), FN.item()

def cal_metric(conf_matrix):
    TP = conf_matrix[1, 1].item()
    TN = conf_matrix[0, 0].item()
    FP = conf_matrix[0, 1].item()
    FN = conf_matrix[1, 0].item()
    return TP, TN, FP, FN

def cal_specificity(TN, FP):
    return TN / (TN + FP)

def cal_sensitivity(TP, FN):
    return TP / (TP + FN)

def cal_accuracy(TP, TN, num_samples):
    return (TP + TN) / num_samples

def cal_cohen_kappa(TP, TN, FP, FN):
    num_samples = TP + TN + FP + FN
    p_o = (TP + TN) / num_samples
    p_neg = ((TN + FP) / num_samples) * ((TN + FN) / num_samples)
    p_pos = ((FN + TP) / num_samples) * ((FP + TP) / num_samples)
    p_e = p_neg + p_pos
    return (p_o - p_e) / (1 - p_e)

if __name__ == '__main__':
    args = parser.parse_args()
    # setup the model
    model = ViT('B_16_imagenet1k', pretrained=False, num_classes=2)
    model.load_state_dict(torch.load('{}/model.pt'.format(args.ckpt_path), map_location='cpu'))
    if args.cuda:
        model.cuda()
    # val set
    test_set = ReactDataset(args.data_path, '{}/d8b_gtruth_clean.csv'.format(args.data_path), image_size=model.image_size, mode='test')
    test_loader = DataLoader(dataset=test_set, batch_size=8, shuffle=False, num_workers=0)
    # start to train
    model.eval()
    acc_test = 0
    conf_matrix = torch.zeros(2, 2)
    for batch_idx, (batch_data, batch_target) in tqdm(enumerate(test_loader)):
        batch_data = batch_data.to('cuda' if args.cuda else 'cpu')
        batch_target = batch_target.to('cuda' if args.cuda else 'cpu')
        with torch.no_grad():
            batch_pred = model(batch_data)
            # get the pred label
            batch_pred = torch.argmax(batch_pred, dim=1)
            for t, p in zip(batch_target, batch_pred):
                conf_matrix[t, p] += 1
    # cal the metric
    TP, TN, FP, FN = cal_metric(conf_matrix)
    specificity = cal_specificity(TN, FP)
    sensitivity = cal_sensitivity(TP, FN)
    accuracy = cal_accuracy(TP, TN, TP+TN+FP+FN)
    cohen_kappa = cal_cohen_kappa(TP, TN, FP, FN)
    # print the calculated metrics
    print('[{}] specificity is: {:.3f}'.format(datetime.now(), specificity))
    print('[{}] sensitivity, is: {:.3f}'.format(datetime.now(), sensitivity))
    print('[{}] accuracy is: {:.3f}'.format(datetime.now(), accuracy))
    print('[{}] cohens_kappa is: {:.3f}'.format(datetime.now(), cohen_kappa))