import matplotlib.pyplot as plt 
import numpy as np
import torch

""" First create our input training data: """
x_values=np.array( [[2.7], [2.1], [3.9], [4.7],
                    [5.3], [5.9], [6.0], [9.1],
                    [8.0], [0.6], [3.3], [8.2]],
                   dtype=np.float32)                   

y_values=np.array( [[1.7], [1.4], [3.3], [4.1],
                    [5.2], [6.8], [4.0], [8.8],
                    [6.0], [0.5], [4.3], [8.1]],
                   dtype=np.float32)

""" Next let's create corresponding torch vectors for this imput data:"""
Y_torch = torch.from_numpy(y_values)
X_torch = torch.from_numpy(x_values)

""" Define our learning rate. Durring the backward pass the gradient is multiplied
    by the learning rate to tweak the input just this much. """
learning_rate = 1e-5

""" Now let's define our very simple network of just a single node: """

#  >--O--< 
input_size =1
hidden_size=1
output_size=1

w1 = torch.rand(input_size,
                hidden_size,
                requires_grad=True)

w2 = torch.zeros(hidden_size,
                output_size,
                requires_grad=True)

# Validate our little network:
print(w1)
# torch.Size([1, 1])

print(w2)
# torch.Size([1, 1])

