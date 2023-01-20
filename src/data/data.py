import os

import numpy as np
import torch
from torchvision.transforms import transforms


def mnist():
    # exchange with the corrupted mnist dataset
   

    images = []
    labels = []

    path='C:/Users/Frede/Documents/GitHub/MLOps/data/'

    test = np.load(path+"test.npz")

    myfiles = [myfile for myfile in os.listdir('./data') if myfile.startswith("train")]

    for i in myfiles:
        with np.load(path+i) as data:
            images.extend(data['images'])
            labels.extend(data['labels'])

    train = list(zip(images, labels))
    test = list(zip(test['images'], test['labels']))

    trainloader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)

    return trainloader, testloader

