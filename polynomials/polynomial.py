#import matplotlib.pyplot as plt
import numpy as np
import tools
import torch

from tools import random_poly_sample, shout, make_plot, get_y_value
""" Previously, we trained a model to transform an input by a factor of phi.
    Here we will create a more complex network to fit a polynomial. Then we
    figure out how to plot it. """

""" Instead of creating tensors of training data, we will use random values
    generated from a given polynomial function. """


################################################################################
print("Start")

""" Before we used the variables x and y to represent the training input and
    output. We also defined x = {1-11} and y as a transformation proportional
    to phi. """

""" Here we will use randomly generated x and y values defined from a given
    function to generate our training data. """
# 1, -40, -100, 1, 0
f = {
    "x^4" :    1,
    "x^3" :  -40,
    "x^2" : -100,
    "x"   :    1,
    "c"   :    0
}
input_size  = 1
hidden_size = 10
output_size = 1

learning_rate = 1e-7
w1 = torch.rand(input_size, hidden_size, dtype=torch.float64, requires_grad=True)
wh1 = torch.rand(hidden_size, hidden_size, dtype=torch.float64, requires_grad=True)
wh2 = torch.rand(hidden_size, hidden_size, dtype=torch.float64, requires_grad=True)
w2 = torch.rand(hidden_size, output_size, dtype=torch.float64, requires_grad=True)

y_pred = ""

for i in range(1, 200000):
    xT, yT = random_poly_sample(f['x^4'],
                                f['x^3'],
                                f['x^2'],
                                f['x'],
                                f['c'])

    # perform the forward pass
    y_pred = xT.mm(w1).mm(wh1).mm(wh2).mm(w2)

    # calculate the error
    loss = (y_pred - yT).pow(2).sum()

    # Perform the backward pass
    loss.backward()

    if i < 20 or i==500 or i==5000 or i==50000:
        shout(i, y_pred, w1, wh1, w2)

    with torch.no_grad():
        w1  -= learning_rate * w1 .grad - learning_rate
        wh1 -= learning_rate * wh1.grad - learning_rate
        wh2 -= learning_rate * wh2.grad - learning_rate
        w2  -= learning_rate * w2 .grad - learning_rate
        w1.grad.zero_()
        wh1.grad.zero_()
        wh2.grad.zero_()
        w2.grad.zero_()


plt = make_plot(f['x^4'], f['x^3'], f['x^2'], f['x'], f['c'])

x = np.linspace(tools.MIN, tools.MAX, 256, endpoint = True)
y = np.empty([256])
#print("x is a: ", y)
#print("x is a: ", x)

i=0
for k in x:
    #y[i]= get_y_value(f['x^4'], f['x^3'], f['x^2'], f['x'], f['c'], k)
    # Use the model to make a prediction:
    xK = torch.tensor([[k]], dtype=torch.float64)
    y_pred = xK.mm(w1).mm(wh1).mm(wh2).mm(w2)
    y[i]=y_pred.detach().numpy()
    i=i+1

#print("Here is y again:", y)
plt.plot(x, y, label = 'prediction')
plt.show()
