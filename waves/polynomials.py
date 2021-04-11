#import matplotlib.pyplot as plt
import numpy as np
import data
import torch

############################################################

import torch.nn as nn
import torch.optim as optim


class Polynomial(nn.Module):

    def __init__(self, input_size  = 256, hidden_size = 16, output_size = 1):
        super(Polynomial, self).__init__()

        # nn.Linear applies a linear tranforation st: y = xAT + b
        # Use nn.Linear to create 3 fully connected layers:
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    # override the forward pass:
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)

        # Recall that "log(softmax(x))" is unstable.
        return F.log_softmax(x, dim=-1)


if __name__ == '__main__':
    print("Welcome to Polynomial land.")
    # First instantiate our model,
    waves = Polynomial()

    # next create an optimizer,
    optimizer = optim.Adam(waves.parameters())

    # then pick loss function,
    loss_fn=nn.NLLLoss()

    #instantiate Xtrain_[256] and Ytrain_


    predictions = []
    epochs=10
    i=0
    # run through some epochs
    for epoch in range(1, epochs):
        #get_training_data:

        #sample_to_one_hot()
        if i == 0:
            print("Check 6")
        #zero out the gradients
        optimizer.zero_grad()
        #  … do stuff …

        Zpred = waves(Xtrain_)

        loss = loss_fn(Zpred, Ztrain_)
        loss.backward()


        optimizer.step()

        if i == 0:
            print("Check 7")
        i=i+1
    # finally print our plot.
    data.plot_waves().show()
