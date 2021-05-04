import matplotlib.pyplot as plt
import numpy as np
import torch
import tools
from tools import yell
from torch import nn

""" In this app, we train a very simple neural network to output zero no matter
the input."""

""" Start by creating our model. Then create an Optimizer and an training loop.
"""

class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(1, 1))
        self.bias   = nn.Parameter(torch.rand(1))

    def forward(self, input):
        return (input @ self.weight) + self.bias


m = Regressor()
learning_rate = 1e-4
optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)
i = torch.rand(1)
print("\nHere is our sample input: ", i)
print(  "Here is the result from our network: ", m(i))
print(  "Here are our parameters: ")

for p in m.named_parameters():
    print(p)

for i in range(1000):
    input       = torch.rand(1)
    null_tensor = torch.zeros(1)
    output      = m(input)
    loss        = torch.abs(output)
    m.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        yell(input, m.weight, m.bias, output, loss)

print("\n\nDid we get this far? Here is m again: ")
for p in m.named_parameters():
    print(p)
