from data import get_mnist_data
from modelarc import modelarc
import torch

EPOCHS = 30

trainloader, testloader = get_mnist_data()

print(trainloader.dataset.data.shape)
print(trainloader.dataset.targets.shape)

for images, labels in trainloader:
    print(images.shape)
    print(labels.shape)
    break

model = modelarc(28*28)