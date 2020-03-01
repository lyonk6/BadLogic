import pandas as pd
import numpy as np
import torch
import generate_training_data

training_data = generate_training_data.training_data


# Define our layers:
input = torch.rand(3, 1).requires_grad_()
h1 = torch.rand(1, 3).requires_grad_()
h2 = torch.mm(input, h1).requires_grad_()
h3 = torch.rand(3, 1).requires_grad_()
output = torch.mm(h2, h3)
#Recall that if A is an m × n matrix and B is an n × p matrix:

print(input)
print(h1)
print(h2)


# print(training_data)
