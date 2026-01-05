import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import torchvision.transforms.functional as F
#steps to training, for one epoch
# 1. zero out the gradients of the optimizer
# 2. Fill `outputs` with the model's predictions for the current `inputs`.
# 3. Calculate the loss using the `loss_function` with `outputs` and `targets`.
# 4. Perform backpropagation with `loss.backward()` and update the model parameters using the `optimizer`.

def train_epoch(model, loss_function, optimizer, train_loader, device):

    model.to(device)
    model.train()

    running_loss = 0.0
    # Initialize the number of correct predictions to 0
    num_correct_predictions = 0
    # Initialize the total number of predictions to 0
    total_predictions = 0
    EPOCH = 1
    for i in range(EPOCH):
        # Print the first 5 indices and labels from the train_loader (only for the first epoch and first batch)
        if i == 0:
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                print("First 5 indices:", list(range(5)))
                print("First 5 labels:", targets[:5].tolist())
                break
        # Iterate over the batches of data from the training DataLoader
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move the inputs and targets to the specified device
            inputs, targets = inputs.to(device), targets.to(device)

            # Shift target labels down by 1 (adjusting for EMNIST letters dataset)
            targets = targets - 1

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_function(outputs, targets)

            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            # Accuracy
            # Get the predicted indices (by taking the argmax along dimension 1 of the outputs).
            predicted_indices = outputs.argmax(dim=1) # rows are the images, column are the scores so we get the max score for each img, ex. A B C. max would give you the score, argmax give you the letter which had max score
            print("First 5 predicted indices:", predicted_indices[:5].tolist())
            print("First 5 target labels:", targets[:5].tolist())
            # Compare predicted indices to actual targets
            correct_predictions = predicted_indices.eq(targets)

            # Sum of correct predictions in the current batch.
            num_correct_in_batch = predicted_indices.eq(targets).sum().item()
            print("First 5 correct predictions in batch (bool):", correct_predictions[:5].tolist())
            print("num_correct_in_batch:", num_correct_in_batch)
            # Add correct predictions to the total correct predictions.
            num_correct_predictions += num_correct_in_batch
            print("num_correct_predictions so far:", num_correct_predictions)
            # Get the batch size from the targets and add it to total predictions.
            batch_size = targets.size(0) # cant hardcode 32 because the last batch may have less. targets is just a 1 dim array
            print("batch_size:", batch_size)
            total_predictions += batch_size
            print("total_predictions so far:", total_predictions)

        # Calculate the average loss for the epoch. 
        # Divide the running loss by the number (len of train loader) of batches.
        average_loss = running_loss / len(train_loader)

        # Calculate the accuracy percentage for the epoch. Multiply correct predictions by 100.
        accuracy_percentage = (num_correct_predictions / total_predictions) * 100


        # Conditionally print based on verbose flag
        print(
            f"Epoch Loss (Avg): {average_loss:.3f} | Epoch Acc: {accuracy_percentage:.2f}%"
        )

    # Return the trained model and average loss
    return model