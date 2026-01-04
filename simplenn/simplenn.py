import pandas as pd
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from modelarc import model
from train import train_epoch
# data ingestion 

# data preparation
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
mean = 0.1722273074089701
std = 0.33094662810661823
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_dataset = datasets.EMNIST(
    root = "./data",
    transform=transform,
    train = True,
    download = True,
    split = 'letters'
)
train_dataloader = DataLoader(
    dataset = train_dataset,
    batch_size = 64,
    shuffle= True
)
test_dataset = datasets.EMNIST(
    root = "./data",
    transform = transform,
    train = False,
    download = True,
    split = 'letters'
)
test_dataloader = DataLoader(
    dataset = train_dataset,
    batch_size = 64,
    shuffle= False
)

# for i in train_dataloader:
#     print(i)



# model archtecture
mainmodel, loss_function, optimizer = model()
# training

model_one_train_epoch, _ = train_epoch(model=mainmodel, # model
                                       loss_function=loss_function, # loss_function
                                       optimizer=optimizer, # optimizer
                                       train_loader=train_dataloader, # train_loader
                                       device=DEVICE) # DEVICE
# eval