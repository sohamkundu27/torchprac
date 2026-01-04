import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True) # wraps the dataset, iterate in batches of 64. pin_memory speeds up GPU transfer, harmless on CPU

    n_pixels = 0 # total pixels
    channel_sum = 0.0 # sum of pixel vals
    channel_sum_sq = 0.0 # sum of squared pixel vals, needed for variance

    for x, _ in loader: # we dont need labels
        b, c, h, w = x.shape #b is batch size, c is channels (1 for EMNIST), h/w is height width
        pixels = b * h * w

        channel_sum += x.sum().item() #	x.sum(): sum of all pixel values in this batch. .item(): convert single-value tensor → Python float.
        channel_sum_sq += (x * x).sum().item()
        n_pixels += pixels

    mean = channel_sum / n_pixels
    var = (channel_sum_sq / n_pixels) - (mean * mean)
    std = var ** 0.5 # std is just sqrt of the var
    return mean, std

train_ds = datasets.EMNIST(
    root="./data",
    split="letters",     
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

mean, std = compute_mean_std(train_ds)
print("mean:", mean, "std:", std)
