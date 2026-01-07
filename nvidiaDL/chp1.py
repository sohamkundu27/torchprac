import random
# perceptron function 
# Sign or Signum function 
# takes in 2 mats, this is a dot product between, so there is one weight 
random.seed(42)
def compute_output(w, x):
    z = 0
    for i in range(len(w)):
        z += x[i] * w[i]
    return 1 if z >= 0 else -1

# A list of input VECTORS (each has 3 features for example)
X_train = [
    [1,  1, -1],
    [1, -1,  1],
    [-1, 1,  1]
]

# A list of TARGETS (one for each vector)
Y_train = [1, -1, 1]

# Weights (same length as a single input vector)
w = [0.1, 0.2, -0.5]
lr = 0.1
# print(compute_output(w,x))
# takes a list of features(X)
def perceptron_training(w, X, Y, lr):
    all_correct = False
    while not all_correct:
        all_correct = True
        
        # Iterate through the SAMPLES (the rows of data), not the indices of weights
        for i in range(len(X)):
            current_input = X[i]
            target = Y[i]
            
            # Predict using the current input vector
            pred = compute_output(w, current_input)
            
            if pred != target:
                all_correct = False
                # Update ALL weights based on this error
                # We need to loop through weights to update them
                for j in range(len(w)):
                    w[j] += lr * target * current_input[j]
                    
    return w

print(perceptron_training(w, X_train, Y_train, lr))
