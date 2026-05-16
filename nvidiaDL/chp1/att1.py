# Open a blank file and implement a perceptron training loop, including data handling, convergence logic, and visualization, with no references.
import numpy as np
# compute the output of a perceptron(single neuron, can be multiple input features) - summation of all the x's(features) and their corresponding weights
#output is z, this is a sign function so the output will be either -1, if negetive and 1 if positive 
def compute_output(w, x, b):
    z = np.dot(w, x) + b
    return np.sign(z)
