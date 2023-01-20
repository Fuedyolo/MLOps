
import click
import wandb
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from model import MyAwesomeModel
from src.data.data import mnist

def train(lr=0.001):
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
    plt.savefig("reports/figures/lossplot.png")

    checkpoint = {
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, "models/checkpoint.pth")
    print('success')