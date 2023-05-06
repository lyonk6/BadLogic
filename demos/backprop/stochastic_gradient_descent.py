import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

MIN = 1
MAX = 100
# Note that numpy.random.rand generates a uniform distribution
# while numpy.random.randn generates a normal distribution.
X = 2 * np.random.rand(MAX, MIN)
y = 4 + 3 * X + np.random.randn(MAX, MIN)
X_b = np.c_[np.ones((MAX, MIN)), X]
#print("This is what we get for X: ", X)
#print("This is what we get for y: ", y)


def batch_gradient_descent():
    eta = 0.1
    n_interations = 1000
    m=100
    theta = np.random.randn(2,1)

    for iteration in range(n_interations):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) -y)
        theta = theta - eta * gradients

    print("Here be the normal: ", theta)

def stochastic_gradient_descent():
    n_epochs = 50
    m = 100
    # Learning schedule hyperparameters (Recall that
    # hyperparameters control learning process):
    t0, t1 = 5, 50

    def learning_schedule(t):
        return t0 / (t + t1)

    theta = np.random.randn(2,1)

    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradient = 2 * xi.T.dot(xi.dot(theta)-yi)
            eta      = learning_schedule(epoch*m+1)
            theta    = theta - eta * gradient

    print("Okay so for SGD we have theta =  ", theta)

# Use a perceptron from SKlearn:
def perceive_flowers():
    iris = load_iris()
    X = iris.data[:, (2,3)]
    y = (iris.target == 0).astype(np.int32)

    per_clf = Perceptron()
    per_clf.fit(X, y)

    y_pred = per_clf.predict([[2, 0.5]])
    print("Here is the inputs : ", X)
    print("Here is the outputs: ", y)

if __name__ == "__main__":
    batch_gradient_descent()
    print("1/3")
    stochastic_gradient_descent()
    print("1/3")
    perceive_flowers()
    print("Done")