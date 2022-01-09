import numpy as np 
import argparse
import torch
from vit_models import ViT
from dataset_loader import ReactDataset
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from datetime import datetime
import os

"""
this script is used to train the network using ViT 
"""

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='./dataset', help='dataset path')
parser.add_argument('--batch-size', type=int, default=8, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='the initial learning rate')
parser.add_argument('--train-steps', type=int, default=20000, help='total training steps')
parser.add_argument('--warmup-steps', type=int, default=500, help='learning rate warm up steps')
parser.add_argument('--ckpt-path', type=str, default='saved_ckpt', help='ckpt path')
parser.add_argument('--cuda', action='store_true', help='if use cuda')

"""
need to optimize the accuracy metric
"""

def accuracy(output, target, mode='train'):
    """Computes the precision@k for the specified values of k"""""
    batch_size = target.size(0)
    pred = torch.argmax(output, dim=1)
    correct = pred.eq(target)
    correct = correct.float().sum()
    if mode == 'train':
        correct = correct / batch_size
    return correct

if __name__ == '__main__':
    args = parser.parse_args()
    # create the folder
    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path, exist_ok=True)
    # setup the model
    model = ViT('B_16_imagenet1k', pretrained=True, num_classes=2)
    if args.cuda:
        model.cuda()
    # load the dataset
    train_set = ReactDataset(args.data_path, '{}/d8a_gtruth_clean.csv'.format(args.data_path), image_size=model.image_size)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # val set
    test_set = ReactDataset(args.data_path, '{}/d8b_gtruth_clean.csv'.format(args.data_path), image_size=model.image_size, mode='test')
    test_loader = DataLoader(dataset=test_set, batch_size=8, shuffle=False, num_workers=0)
    # start to train
    criterion = nn.CrossEntropyLoss()
    # other params
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
        momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=args.lr,
        pct_start=args.warmup_steps / args.train_steps,
        total_steps=args.train_steps)
    # best acc
    best_acc = 0
    # start the training
    epochs = args.train_steps // len(train_loader)
    for epoch_id in range(epochs):
        model.train()
        for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
            batch_data = batch_data.to('cuda' if args.cuda else 'cpu')
            batch_target = batch_target.to('cuda' if args.cuda else 'cpu')
            batch_pred = model(batch_data)
            loss = criterion(batch_pred, batch_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_acc = accuracy(batch_pred, batch_target)
            if batch_idx % 100 == 0:
                print('[{}] epoch: {}/{}, iter: {}/{}, acc_batch: {:.3f}, loss: {:.4f}, acc_best: {:.3f}'.format(datetime.now(), epoch_id+1, epochs+1, \
                                                                        batch_idx+1, len(train_loader)+1, train_acc.item(), loss.item(), best_acc))
        # start to eval the model
        if epoch_id % 1 == 0:
            print('[{}] start to val the model'.format(datetime.now()))
            model.eval()
            test_acc = 0
            for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
                batch_data = batch_data.to('cuda')
                batch_target = batch_target.to('cuda')
                with torch.no_grad():
                    batch_pred = model(batch_data)
                    acc_tmp = accuracy(batch_pred, batch_target, mode='test')
                acc_tmp = acc_tmp.item()
                test_acc += acc_tmp
            test_acc = test_acc / len(test_set)
            # record
            if test_acc > best_acc:
                best_acc = test_acc                                            
                torch.save(model.state_dict(), '{}/model.pt'.format(args.ckpt_path))