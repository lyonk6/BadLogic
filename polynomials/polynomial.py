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


#for i in range(1, 2000):
