import torch
import torch.nn as nn

def evaluate(model, testloader, device):


    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for X, y in testloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            predicted = pred.argmax(dim=1) # go from (64, 10) to (64), argmax is take the highest arg
            correct += (predicted == y).sum().item() # .sum is count the trues, item is tensor to scalar(python int)
            total += y.size(0)
        print(correct/total)
        

    return total


