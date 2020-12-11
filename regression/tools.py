import matplotlib.pyplot as plt
import numpy as np

def plot_regression(x, y, v=[0], plot_line=False):
    """ Next plot the data as a scatterplot and add our regression line too. """
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, label='Original data', s=250, c='g')
    if plot_line and len(v) == len(x):
        plt.plot(x, v, label = 'Fitted line')
    plt.legend()
    plt.show()

def shout(p, yp, v1, v2):
    print("shout ", p)
    print("  y_pred is:\n", yp)
    print("  Here is v1: ", v1)
    print("  Here is v2: ", v2)
    #print("  Here is v2: ", v2)
    print("")
