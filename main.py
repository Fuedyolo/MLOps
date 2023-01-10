import argparse
import sys

import click
import wandb
import matplotlib.pyplot as plt
import torch
from torch import nn, optim

from data import mnist
from model import MyAwesomeModel


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    print("Training day and night")
    print(lr)
    wandb.init(
    project="my-awesome-project",
    config={
    "learning_rate": lr,
    }
)

    model = MyAwesomeModel()
    trainloader, _ = mnist()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epochs = 20
    images, labels = next(iter(trainloader))
    loss_func = []
    counter = 0 
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            counter +=1

            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            if counter % 10 == 0:
                wandb.log({"loss": loss})
            running_loss += loss.item()
        loss_func.append(running_loss / len(trainloader))
        print(f"Epoch {e+1} loss: {loss_func[e]}")
    wandb.log({"examples" : [wandb.Image(im) for im in images[0:3]]})
    plt.plot(loss_func)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    checkpoint = {
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, "./checkpoint.pth")


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    checkpoint = torch.load(model_checkpoint)
    model = MyAwesomeModel()
    model.load_state_dict(checkpoint["state_dict"])
    _, test_set = mnist()
    criterion = nn.NLLLoss()
    accuracy = 0
    loss = 0
    _, testloader = mnist()

    for images, labels in testloader:
        output = model.forward(images)

        labels = labels.type(torch.LongTensor)
        loss += criterion(output, labels).item()

        ps = torch.exp(output)

        temp = labels.data == ps.max(1)[1]

        accuracy += temp.type_as(torch.FloatTensor()).mean()

        print(
            "Test Loss: {:.3f}.. ".format(loss / len(testloader)),
            "Test Accuracy: {:.3f}".format(accuracy / len(testloader)),
        )


cli.add_command(train)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
