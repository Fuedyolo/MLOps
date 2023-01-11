import pytest
from tests import _PATH_DATA
import torch
import numpy as np
import os


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data():
   dataset = torch.load(_PATH_DATA+'/processed/train.pt')

   assert len(dataset) == 25000 
   #assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format
   images, labels = next(iter(dataset))
   assert len(np.unique(labels)) == 10

def func(x):
    return x + 1


def test_answer():
    assert func(3) == 5
