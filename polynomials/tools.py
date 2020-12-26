import matplotlib.pyplot as plt
import numpy as np
import torch
import random

MIN = -6
MAX =  6
IMAGE_WIDTH = 256
# MAX-MIN / IMAGE_WIDTH = 0.046875

# Return a random x, y Pair
def random_poly_sample(a, b, c, d, e):
    x = random.uniform(MIN, MAX)
    x = 6 - 0.046875 * (5)
    y = get_y_value(a, b, c, d, e, x)
    return torch.tensor([[x]], dtype=torch.float64), torch.tensor([[y]], dtype=torch.float64)
    #return x, y

def wide_poly_sample(a, b, c, d, e):
    pxT, yT = random_poly_sample(a, b, c, d, e)
    xT = torch.zeros((1, IMAGE_WIDTH), dtype=torch.float64)
    x = pxT[0][0]
    adjustment = (IMAGE_WIDTH)/(MAX - MIN)
    position = int(torch.floor((adjustment * (x + 6)) + 0.5).item())
    xT[0][position]=1

    return xT, yT

# Return a random x, y Pair
def get_y_value(a, b, c, d, e, x):
    return (a * (x * x * x * x)) + (b * (x * x * x)) + (c * (x * x)) + (d * x) + e

def make_plot(a, b, c, d, e):
    # Recall linspace can be used to create a np.array of evenly
    # spaced integers: numpy.linspace(start, stop, num=50 ...)
    #x = np.linspace(0, 10, IMAGE_WIDTH, endpoint = True)
    x = np.linspace(MIN, MAX, IMAGE_WIDTH, endpoint = True)
    y = (a * (x * x * x * x)) + (b * (x * x * x)) + (c * (x * x)) + (d * x) + e



    plt.plot(x, y, '-g', label=r'$y = ax^2 + bx + c$')
    return plt
    #plt.show()



def shout(p, yp, v1, vh, v2):
    print("shout ", p)
    print("  y_pred is:\n", yp)
    print("  Here is v1: ", v1)
    print("  Here is vh: ", vh)
    print("  Here is v2: ", v2)
    print("")


if __name__ == '__main__':
    a, b = random_poly_sample(0, 0, 2, 1, 0)
    #print("Here is a random sample:\n", a, "\n\n", b)
    print("\n")
    aW, bW = wide_poly_sample(0, 0, 2, 1, 0)
    #print("Here is a wide sample:\n", aW, "\n\n", bW)
    print("wide_poly_sample x:", aW)
    print("wide_poly_sample y:", bW)
    #make_plot(1, -40, -100, 1, 0).show()

    ## Curvy ones:
    # make_plot(1, 5, -15, -100, 0)
    # make_plot(1, -9, -25, 100, 0)
    # make_plot(1, -90, 0, 0, 0)
    # make_plot(0.5, -40, 0, 0, 0)
    # make_plot(-0.05, 40, 100, 0, 0)
    # make_plot(1, 40, 100, 0, 0)
    # make_plot(1, -40, -100, 1, 0)
