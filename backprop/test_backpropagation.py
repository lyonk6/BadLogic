import pytest
import numpy as np
import mnist
import backpropagation as bp

def test_one_hot():
    V = np.random.randint(0, 10, 10)
    print("before:", V)
    print("shape: ", V.shape)
    V = bp.one_hot(V)
    print("after:\n", V)
    print("shape: ", V.shape)

def test_ReLU():
    test_array = np.array([-0.2, -0.1, -0.0, 0.1, 0.2])
    response   = bp.ReLU(test_array)
    # [[0.  0.  0.  0.  0.1 0.2 0.2 0.3]]
    assert response[0] == 0.0
    assert response[1] == 0.0
    assert response[2] == 0.0
    assert response[3] == 0.1
    assert response[4] == 0.2

def test_d_ReLU():
    test_array = bp.ReLU(np.array([-0.2, -0.1, -0.0, 0.1, 0.2]))
    response   = bp.d_ReLU(test_array)
    assert response[0] == 0
    assert response[1] == 0
    assert response[2] == 0
    assert response[3] == 1
    assert response[4] == 1

def init_real_parameters():
    w1, b1, w2, b2, _ = init_test_parameters()
    d = mnist.training_data()
    image = d.pop_image()
    X = image[1:]
    label = image[:1]
    print("Shape of real X: ", X.shape)
    return w1, b1, w2, b2, X, label

def init_test_parameters():
    # Balance the weights and biases to make this test easy.
    w1 = np.zeros((10, 10)) + 1/10
    b1 = np.zeros((10, 1))
    w2 = np.zeros((10, 10)) + 1/10
    b2 = np.zeros((10, 1))
    X =  np.zeros((10, 1)) + 1
    print("Shape of artificial X: ", X.shape)
    return w1, b1, w2, b2, X

def test_forward_pass():
    # Easy Test:
    w1, b1, w2, b2, X = init_test_parameters()
    Z1, A1, Z2, A2  = bp.forward_pass(w1, b1, w2, b2, X)
    assert np.array_equal(Z1, Z2)
    assert np.array_equal(A1, Z1)

def test_backward_pass():
    init_real_parameters()
    w1, b1, w2, b2, X = init_test_parameters()
    Z1, A1, Z2, A2  = bp.forward_pass(w1, b1, w2, b2, X)
    Y = np.random.randint(0, 10, 10)
    print(Y)
    """
    dW1, db1, dW2, db2 = bp.backward_pass(Z1, A1, Z2, A2, w2, X, Y)
    print("dW1", dW1)
    print("db1", db1)
    print("dW2", dW2)
    print("db2", db2)
    #"""
    # Recall the cost function is: (A2 - Y)^2
    #Z1, A1, Z2, A2, w2, X, Y
    assert True
