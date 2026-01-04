# model architecture will give the model, loss function, and the optimizer 
import torch
import torch.nn as nn
from torch import optim
def model():
    model = nn.Sequential(
        nn.Flatten(),

        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128,26)
    )

    loss_function = nn.CrossEntropyLoss() # use for multi class classification

    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    
    return model, loss_function, optimizer
