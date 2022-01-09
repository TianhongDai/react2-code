import numpy as np 
from PIL import Image
import torch
import csv
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

"""
this is the file to load the dataset of React2
"""

class ReactDataset(Dataset):
    def __init__(self, data_path, csv_path, image_size, mode='train'):
        self.data_path = data_path
        self.csv_path = csv_path
        self.mode = mode
        # read the name and labels
        self.im_lst, self.labels = [], []
        with open(self.csv_path, 'r') as f:
            csv_reader = csv.reader(f)
            _ = next(csv_reader)
            for row in csv_reader:
                self.im_lst.append(row[-1])
                label = 1 if row[-2] == 'True' else 0
                self.labels.append(label)
            assert len(self.im_lst) == len(self.labels)
        self.prefix = 'd8a_win_clean' if self.mode == 'train' else 'd8b_win_clean'
        self.image_size = image_size
        self.transform = transforms.Compose([transforms.Resize(self.image_size), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),])

    def __getitem__(self, idx):
        # read the image and the target
        im = Image.open('{}/{}/{}'.format(self.data_path, self.prefix, self.im_lst[idx]))
        label = self.labels[idx]
        # do the preprocessing
        im = self.transform(im)
        return im, label

    def __len__(self):
        return len(self.im_lst)


if __name__ == '__main__':
    dataset = ReactDataset('dataset', 'dataset/d8a_gtruth_clean.csv', image_size=(256, 256))
    train_loader = DataLoader(dataset=dataset, batch_size=6, shuffle=False, num_workers=0)
    for i, (im, label) in enumerate(train_loader):
        print(im.shape, label.shape)