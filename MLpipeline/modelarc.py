import torch
import torch.nn as nn
import torch.nn.functional as f

def modelarc(input_size):
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(input_size, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10)
    )
    return model

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2D(
            in_channels = 1, # channels in the input image (grayscale)
            out_channels = 64, #we are learning 64 filters, each producing one feature map(28x28)(each with its own useful pattern).
            kernel_size = 3, # 3x3 grid of learnable weights that slides over the images
            padding = 1 # so the grid can center on the edges
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # halves spatial size: 28x28 -> 14x14



    