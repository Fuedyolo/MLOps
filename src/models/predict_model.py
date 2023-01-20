import torch
from torch import nn

from src.data.data import mnist

from models.model import MyAwesomeModel


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