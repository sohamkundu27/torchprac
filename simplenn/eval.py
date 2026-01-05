import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import torchvision.transforms.functional as F

def eval(model,test_loader, device):

    model.to(device)
    model.eval()
    running_loss = 0
    num_correct_predictions = 0
    total_predictions = 0 
    with torch.no_grad():
    # Iterate over the batches of data from the training DataLoader
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            # Move the inputs and targets to the specified device
            inputs, targets = inputs.to(device), targets.to(device)

            # Shift target labels down by 1 (adjusting for EMNIST letters dataset)
            targets = targets - 1


            outputs = model(inputs)


            # Accuracy
            # Get the predicted indices (by taking the argmax along dimension 1 of the outputs).
            predicted_indices = outputs.argmax(dim=1) # rows are the images, column are the scores so we get the max score for each img, ex. A B C. max would give you the score, argmax give you the letter which had max score

            correct_predictions = predicted_indices.eq(targets)

            # Sum of correct predictions in the current batch.
            num_correct_in_batch = predicted_indices.eq(targets).sum().item()

            num_correct_predictions += num_correct_in_batch

            batch_size = targets.size(0) # cant hardcode 32 because the last batch may have less. targets is just a 1 dim array
            total_predictions += batch_size



            # Calculate the accuracy percentage for the epoch. Multiply correct predictions by 100.
            accuracy_percentage = (num_correct_predictions / total_predictions) * 100


        print(
            f"EVAL- test acc: {accuracy_percentage:.2f}%"
        )

    # Return the trained model and average loss
    return accuracy_percentage