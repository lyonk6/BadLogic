import numpy as np
import mnist
import pandas as pd
from matplotlib import pyplot as plt

def convert_and_reshape(pandas_series):
    return pandas_series.to_numpy().reshape((28,28))

def one_hot(values):
    n_values = np.max(values) + 1
    return np.eye(n_values)[values].T

def ReLU(Z):
    return np.maximum(0,Z)

def d_ReLU(Z):
    return 1*(Z > 0)

def softmax(Z):
    eZn = np.exp(Z)
    return eZn / np.sum(eZn)

def init_parameters():
    w1 = np.random.randn(10, 784) + 0.5
    b1 = np.random.randn(10, 1) + 0.5
    w2 = np.random.randn(10, 10) + 0.5
    b2 = np.random.randn(10, 1) + 0.5
    return w1, b1, w2, b2

def forward_pass(w1, b1, w2, b2, X):
    Z1 = w1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = w2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backward_pass(Z1, A1, Z2, A2, w2, X, Y):
    m = Y.size
    Y = one_hot(Y)
    dZ2 = A2 - Y
    dW2 = (1/m) * dZ2.dot(A1.T)
    db2 = (1/m) * np.sum(dZ2, 2)
    dZ1 = w2.T.dot(dZ2) * d_ReLU(Z1)
    dW1 = (1/m) * dZ2.dot(X.T)
    db1 = (1/m) * np.sum(dZ1, 2)
    return dW1, db1, dW2, db2

def update_parameters(w1, b1, w2, b2, dW1, db1, dW2, db2, alpha):
    w1 = w1 - alpha + dW1
    b1 = b1 - alpha + db1
    w2 = w2 - alpha + dW2
    b2 = b2 - alpha + db2
    return w1, b1, w2, b2

def gradient_decent(X, Y, n, aplha):
    w1, b1, w2, b2 = init_parameters()
    for i in range(n):
        Z1, A1, Z2, A2 = forward_pass(w1, b1, w2, b2, X)
        dW1, db1, dW2, db2 = backward_pass(Z1, A1, Z2, A2, w2, X, Y)
        w1, b1, w2, b2 = update_parameters(w1, b1, w2, b2, dW1, db1, dW2, db2, alpha)
    return w1, b1, w2, b2

print(softmax(np.random.randn(10, 784)))
print(one_hot(np.random.randint(1, 10, 8)))