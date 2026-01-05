import numpy as np

def forward_pass(W, b, X):
        return W @ X + b # @ is matmuls, can use * when there is broadcasting or of the mm's are the same dims

def mse_loss(y_true, y_pred):
        return np.mean((y_pred-y_true) ** 2) # mean because it's a vector, y hat is the model's prediction

def backward_linear_mse():
        return

def train_linear():
        return