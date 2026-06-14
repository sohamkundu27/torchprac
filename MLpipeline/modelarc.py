import torch
import torch.nn as nn
import torch.nn.functional as f

def modelarc(input_size):
    model = torch.nn.(
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
            in_channels = 1,
            out_channels = 64, #we are learning 64 filters, each producing one feature map(each with its own useful pattern).
            kernel_size = 3,
            padding = 1
        )



    