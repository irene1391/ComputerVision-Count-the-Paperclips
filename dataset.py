from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch
import glob
import numpy as np
import os
import csv
import cv2
import random

class imgdataset(data.Dataset):
    def __init__(self, name, label, add_noise):
        self.image_name = name
        self.image_label = label
        self.add_noise = add_noise

    def __getitem__(self, index):
        img = cv2.imread(self.image_name[index])
        label = self.image_label[index]

        # noise
        if self.add_noise:
            if random.randint(0, 1) == 0:
                img = cv2.transpose(img)

            if random.randint(0, 1) == 0:
                img = cv2.flip(img, random.randint(0, 1))

        img = np.array(img.transpose(2, 0, 1), dtype="float32")
        img = img / 255.0 - 0.5

        return img, np.array(label, dtype="int")

    def __len__(self):
        return len(self.image_name)

def get_dataloader(config, phase):
    is_shuffle = phase == 'train'
    add_noise = phase == 'train'
    batch_size = config.batch_size
    if phase == "test":
        batch_size = 1
    name = []
    label = []
    if phase == "train" or phase == "valid":
        reader = csv.reader(open(config.label_path+"train.csv"))
        for v in reader:
            if v[0] == "id":
                continue
            name.append(config.data_path+"clips-{}.png".format(v[0]))
            label.append(eval(v[1]))
        n = int(config.valid_rate * int(len(name)))
        if phase == "train":
            name = name[:n]
            label = label[:n]
        else:
            name = name[n:]
            label = label[n:]
    else:
        reader = csv.reader(open(config.label_path+"test.csv"))
        for v in reader:
            if v[0] == "id":
                continue
            name.append(config.data_path+"clips-{}.png".format(v[0]))
            label.append(0)

    dataloader = DataLoader(dataset=imgdataset(name, label, add_noise),
                            batch_size=batch_size,
                            shuffle=is_shuffle,
                            num_workers=config.num_workers)
    return dataloader
