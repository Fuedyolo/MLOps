# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import torch
from torchvision import transforms
import glob
import numpy as np


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    transformer = transforms.Normalize((0.5,))

    input = input_filepath+'/raw/'
    output = output_filepath+'/processed/'

    files = glob.glob(input+'/train_*.npz')

    train_images = []
    train_labels = []
    
    test_images = []
    test_labels = []


    for i in files:
        with np.load(i) as data:
            train_images.extend(data['images'])
            train_labels.extend(data['labels'])
    
    with np.load(input+'test.npz') as data:
        test_images = data['images']
        test_labels = data['labels']

    train = list(zip(train_images, train_labels))
    test = list(zip(test_images, test_labels))

    train = torch.tensor(train)
    test = torch.tensor(test)

    trainloader = torch.utils.data.DataLoader(transformer(train), batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(transformer(test), batch_size=64, shuffle=True)

    torch.save(trainloader, output+'train.pt')
    torch.save(testloader, output+'test.pt')

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables

    main()
