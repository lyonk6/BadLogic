#import matplotlib.pyplot as plt
import numpy as np
import tools
import torch
from torch import nn
from tools import wide_poly_sample, make_plot

################################################################################
f = {
    "x^4" :    1,
    "x^3" :  -40,
    "x^2" : -100,
    "x"   :    1,
    "c"   :    0
}

""" Before we used the variables x and y to represent the training input and
    output. We also defined x = {1-11} and y as a transformation proportional
    to phi. """

""" Here we will use randomly generated x and y values defined from a given
    function to generate our training data. """
# 1, -40, -100, 1, 0

class Polynomial(nn.Module):
    def __init__(self, input_size=256, hidden_size=4, output_size=1):
        super().__init__()
        self.input_layer   = nn.Parameter(torch.rand(input_size, hidden_size, dtype=torch.float64))
        self.hidden_layer  = nn.Parameter(torch.rand(hidden_size, hidden_size, dtype=torch.float64))
        self.output_layer  = nn.Parameter(torch.rand(hidden_size, output_size, dtype=torch.float64))

        # What are our inputs?

    def forward(self, input):
        return (input @ self.input_layer @ self.hidden_layer @ self.output_layer)

if __name__ == '__main__':
    print("Start")

    # Part 1:
    ## { Create Model, Create Optimizer, Create Loop}
    model = Polynomial()
    optimizer = torch.optim.Adam(model.parameters(), lr= 1e-4)
    
    for i in range(1000):
        # Part 2:
        ## {forward pass, loss, backward pass}
        xT, yT = wide_poly_sample(f['x^4'], f['x^3'], f['x^2'], f['x'], f['c'])
        output = model(xT)
        loss   = torch.abs(output)

        # part 3:
        ## {zero_grad(), loss.backward(), optimizer.step()}
        model.zero_grad()
        loss.backward()
        optimizer.step()

        # Print an update
        if i % 250 == 0:
            print("\n{} iterations".format(i))
            print("  Loss       = {} ".format(loss))
            print("  Output     = {} ".format(output))
            print("hidden layer = \n{} ".format(model.hidden_layer))
            print("hidden grad  = \n{} ".format(model.hidden_layer.grad))

    x = np.linspace(tools.MIN, tools.MAX, tools.IMAGE_WIDTH, endpoint = True)
    y = np.zeros([tools.IMAGE_WIDTH])
    plt = make_plot(f['x^4'], f['x^3'], f['x^2'], f['x'], f['c'])
    plt.plot(x, y, label = 'prediction')
    plt.show()
