import random
# perceptron function
# Sign or Signum function 
def compute_output(w, x):
    z = 0
    for i in range(len(w)):
        z += x[i] * w[i]
    return 1 if z >= 0 else -1

w = [0.9,0.6,0.5]
x = [1,1,-1]
y = [-1, -1, -1] # actual 
lr = 0.01
print(compute_output(w,x))

def perceptron_training(w, x, y, lr):
    all_correct = False
    while not all_correct:
        all_correct = True
        
        indices = [i for i in range(len(w))]
        random.shuffle(indices)
        for i in indices:
            pred = compute_output(w, x)
            if pred != y[i]:
                w[i] += float(y[i]) * lr * float(x[i]) 
                all_correct = False
        return w

print(perceptron_training(w, x, y, lr))
