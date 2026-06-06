import torch

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


    