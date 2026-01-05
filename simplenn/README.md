# SimpleNN

A simple neural network implementation for classifying EMNIST letters using PyTorch.

## Overview

This project implements a basic feedforward neural network to classify handwritten letters from the EMNIST torchvision dataset. The model consists of three fully connected layers with ReLU activations and is trained using the Adam optimizer.

## Features

- Simple feedforward neural network architecture
- EMNIST letters dataset classification (26 classes: A-Z)
- Data normalization using computed mean and standard deviation
- Training and evaluation scripts
- PyTorch-based implementation

## Project Structure

```
simplenn/
├── simplenn.py          # Main script for data loading, training, and evaluation
├── modelarc.py          # Neural network model architecture
├── train.py             # Training function
├── eval.py              # Evaluation function
├── meanstd.py           # Mean and standard deviation computation
├── requirements.txt     # Python dependencies
├── data/                # EMNIST dataset (downloaded automatically)
└── README.md            # This file
```
## ML Pipeline

# Data Ingestion 
Use datasets.EMNISTS to get all the data, split by letters. use download=True to automatically get all the data on your disk

# Data Preparation
Already prepared, this is a torchvision dataset

# model architecture

sequntial neural network
- Input layer: 784 neurons (28x28 flattened images)
- Hidden layer 1: 256 neurons with ReLU activation
- Hidden layer 2: 128 neurons with ReLU activation
- Output layer: 26 neurons (one for each letter A-Z)

# Training
use the dataloader to get batch_size amount of datapoints on working memory
put data(inputs[real inputs that lead to the targets]and targets) and model on device
feed inputs through the model and get the model predictions
calculate loss with loss function, calculate gradient, update gradients.
track loss

# Eval
similar to training, except change model mode to eval, so theres no dropout and mean/std abd calcualated on everything not just the current batch and use torch no grad to stop updating the weight. no loss calculateion here, everything else can be calculated the same

