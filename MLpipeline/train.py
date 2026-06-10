import torch
import torch.nn as nn

def train(epochs, loss_fn, model, optimizer, trainloader, valloader, device):

    model.to(device)
    for i in range(epochs):
        total_loss = 0
        for X, y in trainloader: # shape of X (64, 1, 28, 28), shape of y (64) runs 60,000 samples / 64 batch size    model.to(device)
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad() # clear gradients from the last run
            pred = model(X) # forward pass get the pred
            loss = loss_fn(pred, y) # calculate the loss
            loss.backward() # backpropogation: calculate the gradients for every parameters
            optimizer.step() # update weights using those gradients, both loss and optimizer are updating the grad parameters in model
            total_loss += loss.item() # collect loss from each batch
        print(f"epoch {i} train_loss =  {total_loss/len(trainloader)}") # average loss per batch across one epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in valloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                predicted = pred.argmax(dim=1)
                correct += (predicted==y).sum().item()
                total += y.size(0)
        print(f"epoch {i} val_loss =  {(correct/total)}") # average loss per batch across one epoch





