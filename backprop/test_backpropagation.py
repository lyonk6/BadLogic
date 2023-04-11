import pytest
import numpy as np
import pandas as pd
import backpropagation as bp

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
    print(response)

"""
def test_backward_pass():
    #Z1, A1, Z2, A2, w2, X, Y
def test_convert_and_reshape():
    #pandas_series
def test_d_ReLU():
    #Z
def test_forward_pass():
    w1, b1, w2, b2, X
def test_init_parameters():
def test_one_hot():
    #V
def test_softmax():
    #Z
"""