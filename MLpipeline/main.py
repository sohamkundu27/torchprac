from data import get_mnist_data
from modelarc import modelarc
import torch
import torch as nn

EPOCHS = 30

trainloader, testloader = get_mnist_data()

print(trainloader.dataset.data.shape)
print(trainloader.dataset.targets.shape)

for images, labels in trainloader:
    print(images.shape)
    print(labels.shape)
    break

model = modelarc(28*28)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss() # softmax we have 10 options

for i in range(EPOCHS):
    total_loss = 0
        for X, y in train_loader: # shape of X (64, 1, 28, 28), shape of y (64) runs 60,000 samples / 64 batch size
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    print(f"epoch" + i + "loss = " + total_loss/len(train_loader)) # average loss per batch across one epoch


    
    