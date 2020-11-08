import numpy as np
import torch
import tools

""" Previously, we trained a model to transform an input by a factor of phi.
    Here we will create a more complex network to fit a polynomial. Then we
    figure out how to plot it. """

""" Instead of creating tensors of training data, we will use random values
    generated from a given polynomial function. """


################################################################################
print("Start")


learning_rate = 1e-5
w1 = torch.zeros(1, 1, requires_grad=True)
w2 = torch.zeros(1, 1, requires_grad=True)

print("Here is a tensor:       ", w1)
print("Here is another tensor: ", w2)

y_pred = ""
predictions = []


for i in range(1, 2000):
    # grab a bit of training data:


    # perform the forward pass
    y_pred = xT.mm(w1).mm(w2)

    if i == 1:
        print("No fucking way man!")
        print("y_pred is a : ", type(y_pred))
        print("infact, here is y_pred: ", y_pred)
        print("Here is w1: ", w1)
        print("Here is w2: ", w2)
