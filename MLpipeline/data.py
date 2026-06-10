import torch 
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from torchvision import datasets, transforms

# gimme mnist and put it in a dataset variable|
def get_mnist_data(batch_size=64):
    transform = transforms.ToTensor()
    # Download and load the training and test datasets
    full_train = datasets.MNIST(root='', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='', train=False, download=True, transform=transform)

    train_size = 50000
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])
    # dataloaders handle the batch distribution and shuffling of the data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

