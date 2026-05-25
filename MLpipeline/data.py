import pytorch from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import datasets, transforms

# gimme mnist and put it in a dataset variable|
def get_mnist_data(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader