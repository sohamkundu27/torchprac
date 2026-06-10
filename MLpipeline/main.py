from data import get_mnist_data
from modelarc import modelarc
from train import train
from evaluate import evaluate
import torch
import torch.nn as nn

epochs = 30

trainloader, valloader, testloader = get_mnist_data()

model = modelarc(28*28)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss() # softmax we have 10 options

device = "cuda" if torch.cuda.is_available() else "cpu"
train(30, loss_fn, model, optimizer, trainloader, valloader, device)
evaluate(model, testloader, device)
    