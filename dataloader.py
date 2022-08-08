import csv
from sklearn.inspection import PartialDependenceDisplay

from torch.utils.data import Dataset

import os
import torch

import pandas as pd
import numpy as np

import torchvision.transforms as transforms 
from torchvision.utils import save_image
import csv

import json
from collections import Counter

import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from os.path import exists

from config import getopt

def transform_train():
    m16_transform_list = transforms.Compose([
        transforms.RandomAffine((1, 15)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return m16_transform_list

def transform_test():
    m16_transform_list = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    return m16_transform_list  

train_dataset = datasets.CIFAR100('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform_train())
val_dataset= datasets.CIFAR100('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform_test())

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    opt = getopt()

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=False, drop_last=False)

    for i, (X, y) in enumerate(dataloader):
        print(X.shape, y.shape)
        print(X)
        print(y)
        break
