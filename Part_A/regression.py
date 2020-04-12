import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

""" First create our input training data: """
# x_values=np.array( [[2.7], [2.1], [3.9], [4.7],
#                     [5.3], [5.9], [6.0], [9.1],
#                     [8.0], [0.6], [3.3], [8.2]],
#                    dtype=np.float32)
#
# y_values=np.array( [[1.7], [1.4], [3.3], [4.1],
#                     [5.2], [6.8], [4.0], [8.8],
#                     [6.0], [0.5], [4.3], [8.1]],
#                    dtype=np.float32)
#

x_values=np.array( [[1.0],  [2.0],  [3.0],
                    [4.0],  [5.0],  [6.0],
                    [7.0],  [8.0],  [9.0],
                    [10.0], [11.0], [12.0]],
                    dtype=np.float32)


y_values=np.array( [[3.0],  [6.0],  [9.0],
                    [12.0], [15.0], [18.0],
                    [21.0], [24.0], [27.0],
                    [30.0], [33.0], [36.0]],
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

w1 = torch.zeros(input_size,
                hidden_size,
                requires_grad=True)

w2 = torch.zeros(hidden_size,
                output_size,
                requires_grad=True)


""" Now we train our neural network on the training data treating X_torch
    as the input and Y_torch as the desired output.  Note that with a
    learning_rate of 1e-5 it takes at least 1000 iterations to see the
    regression line budge. Consider alernate learning functions.

    Additionally, we will define vectors to track how w1, w2, y_pred
    and the gradients change.

    Finally, note the datatypes of w1, w2, y_pred is <class 'torch.Tensor'>
    so it must be converted back to a numpy array before plotting.
    """
interations=1500
outFile = open('outfile', 'w')
for i in range(1, interations):

    # Perform the forward pass
    y_pred = X_torch.mm(w1).mm(w2)

    # Use the standard error function to calculate the MSE.
    loss = (y_pred - Y_torch).pow(2).sum()


    loss.backward()

    """ Now we actually tweak our model. Since we are just updating our model
        we do not need to track the gradient. So we temporarilly turn off the
        model the gradient tracking with `torch.no_grad()` Then we subtract
        the product of the gradient and the learning rate to get the new
        values for w1 & w2. """
    with torch.no_grad():
        w1 -= learning_rate * w1.grad + learning_rate/10
        w2 -= learning_rate * w2.grad + learning_rate/10
        w1.grad.zero_()
        w2.grad.zero_()


    """
    if i % 100 == 0:
        print(i, loss.item())
        print("y_pred: ", y_pred)
        print("w1: ", w1)
        print("w1.grad: ", w1.grad)"""

print('w1: ', w1)
print('w2: ', w2)


""" Now for some reason we are gonna make a new tensor. """
X_tensor = torch.from_numpy(x_values)

#These are the predicted values from our network:
predicted_from_neuron = X_tensor.mm(w1).mm(w2)
predicted_as_np = predicted_from_neuron.detach().numpy()
print("predicted_from_neuron:", predicted_from_neuron)
print("predicted_as_np: ", predicted_as_np)

print("This is where it's at: ")
for a in predicted_as_np:
    print(a)
    print(type(a))
    outFile.write(a)

outFile.close()

""" Next plot the data as a scatterplot and add our regression line too. """
plt.figure(figsize=(12, 8))
plt.scatter(x_values, y_values, label='Original data', s=250, c='g')
plt.plot(x_values, predicted_as_np, label = 'Fitted line')
plt.legend()
#plt.show()
