import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
np.seterr(all='raise')

def one_hot(values):
    n_values = 10
    return np.eye(10)[values].T

def ReLU(Z):
    return np.maximum(Z, 0)

def d_ReLU(Z):
    return 1*(Z > 0)

def softmax(Z):
    eZn = np.exp(Z)
    return eZn / np.sum(eZn, axis=0)

def init_parameters():
    # We desire a network that is uniformally random and 
    # has a strong bias against activation. 
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2

def forward_pass(w1, b1, w2, b2, X):
    Z1 = w1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = w2.dot(A1) + b2
    A2 = softmax(Z2)
    # The size of Z1, A1, Z2, A2 should be: (10, 41000)
    return Z1, A1, Z2, A2

def backward_pass(Z1, A1, Z2, A2, w2, X, Y):
    m = Y.size
    Y = one_hot(Y)
    dZ2 = A2 - Y
    dW2 = (1/m) * dZ2.dot(A1.T)
    #print("This is dZ2: ", dZ2)
    db2 = (1/m) * np.sum(dZ2)
    dZ1 = w2.T.dot(dZ2) * d_ReLU(Z1)
    dW1 = (1/m) * dZ2.dot(X.T)
    db1 = (1/m) * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_parameters(w1, b1, w2, b2, dW1, db1, dW2, db2, alpha):
    w1 = w1 - alpha * dW1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dW2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, n):
    w1, b1, w2, b2 = init_parameters()
    for i in range(n):
        Z1, A1, Z2, A2 = forward_pass(w1, b1, w2, b2, X)
        dW1, db1, dW2, db2 = backward_pass(Z1, A1, Z2, A2, w2, X, Y)
        w1, b1, w2, b2 = update_parameters(w1, b1, w2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print("Accuracy:", get_accuracy(predictions, Y))
    return w1, b1, w2, b2


if __name__ == "__main__":
    print(softmax(np.random.randn(10, 784)))
    print(one_hot(np.random.randint(1, 10, 8)))
    data = pd.read_csv("data/train.csv")
    data = np.array(data)
    m, n = data.shape

    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.

    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255.
    shape_element = X_train[0].shape # should be (41000, )
    shape_image   = X_train[:, 0].shape # should be (784, )
    image   = X_train[:, 0].reshape((28,28)) # should be (784, )
    print(shape_element)
    print(shape_image)

    # plot the sample
    #fig = plt.figure
    #plt.imshow(image, cmap='gray')
    #plt.show()
    w1, b1, w2, b2 = gradient_descent(X_train, Y_train, 0.1, 500)