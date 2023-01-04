import argparse
import sys

import matplotlib.pyplot as plt
import torch
import click

from data import mnist
from model import MyAwesomeModel
from torch import nn, optim


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    model = MyAwesomeModel()
    trainloader, _ = mnist()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epochs = 5
    images, labels = next(iter(trainloader))
    images.resize_(64, 784)
    loss_func = []

    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        loss_func.append(running_loss / len(trainloader))
    plt.plot(loss_func)
    plt.show()

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)


    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist()


cli.add_command(train)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()


#git branch check