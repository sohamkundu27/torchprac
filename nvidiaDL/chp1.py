
# perceptron function
# Sign or Signum function 
def compute_output(w, x):
    z = 0
    for i in range(len(w)):
        z += x[i] * w[i]
    return 1 if z >= 0 else -1

w = [0.9,0.6,0.5]
x = [1,1,-1]

print(compute_output(w,x))