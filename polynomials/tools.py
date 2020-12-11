import matplotlib.pyplot as plt
import numpy as np
import random

MIN = -6
MAX =  6

# Return a random x, y Pair
def random_poly_sample(a, b, c, d, e):
    x = random.uniform(MIN, MAX)
    y = (a * (x * x * x * x)) + (b * (x * x * x)) + (c * (x * x)) + (d * x) + e

    return x, y

a = random_poly_sample(0, 0, 2, 1, 0)
print("Here is a random sample: ", a)

def make_plot(a, b, c, d, e):
    # Recall linspace can be used to create a np.array of evenly
    # spaced integers: numpy.linspace(start, stop, num=50 ...)
    #x = np.linspace(0, 10, 256, endpoint = True)
    x = np.linspace(MIN, MAX, 256, endpoint = True)
    y = (a * (x * x * x * x)) + (b * (x * x * x)) + (c * (x * x)) + (d * x) + e



    plt.plot(x, y, '-g', label=r'$y = ax^2 + bx + c$')

    plt.show()


if __name__ == '__main__':
    make_plot(1, -40, -100, 1, 0)
    ## Curvy ones:
    # make_plot(1, 5, -15, -100, 0)
    # make_plot(1, -9, -25, 100, 0)
    # make_plot(1, -90, 0, 0, 0)
    # make_plot(0.5, -40, 0, 0, 0)
    # make_plot(-0.05, 40, 100, 0, 0)
    # make_plot(1, 40, 100, 0, 0)
    # make_plot(1, -40, -100, 1, 0)
