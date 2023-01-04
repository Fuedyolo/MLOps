import torch
import numpy as np


def mnist():
    # exchange with the corrupted mnist dataset
    test = np.load("C:/Users/Frede/Documents/DTU/Kandidat/3.Semester/MLOps/s1/final_exercise/test.npz")
    train = np.load("C:/Users/Frede/Documents/DTU/Kandidat/3.Semester/MLOps/s1/final_exercise/train_0.npz")

    train = list(zip(train['images'], train['labels']))
    test = list(zip(test['images'], test['labels']))

    trainloader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)

    return trainloader, testloader


