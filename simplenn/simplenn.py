import pandas as pd
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from modelarc import model
# data ingestion 
file_path = "./cifar-10/trainLabels.csv"
data_df = pd.read_csv(file_path)

print(data_df.shape)
# data preparation

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
    train = True,
    download = True,
    split = 'letters'
)
test_dataloader = DataLoader(
    dataset = train_dataset,
    batch_size = 64,
    shuffle= False
)

for i in train_dataloader:
    print(i)



# model archtecture
mainmodel, loss_function, optimizer = model()
# training
# eval