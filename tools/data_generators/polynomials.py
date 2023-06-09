import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import random

MIN = -10
MAX = 10

# Let a, b, c be constants > 0 for the function:
#  Z = (a**2) * (X**2) + (b**2) * (Y**2) - c
def sample_paraboloid(a = 0.25, b = 0.25, c = 5):
    x = random.uniform(MIN, MAX)
    y = random.uniform(MIN, MAX)
    z = (a**2) * (x**2) + (b**2) * (y**2) - c
    return x, y, z

# Let a, b, c be constants > 0 for the function:
#  Z = (a**2) * (X**2) - (b**2) * (Y**2) - c
def sample_saddle(a = 0.25, b = 0.25, c = 5):
    x = random.uniform(MIN, MAX)
    y = random.uniform(MIN, MAX)
    z = (a**2) * (x**2) - (b**2) * (y**2) - c
    return x, y, z

#Sample from sine wave propogating in 2 dimensions:
def sample_waves():
    x = random.uniform(MIN, MAX)
    y = random.uniform(MIN, MAX)
    r = np.sqrt(x**2 + y**2)
    z = np.sin(r)
    return x, y, z

# Define X, Y and Z vectors suitable for a paraboloid.
def paraboloid(a = 0.25, b = 0.25, c = 5):
    # Define X, Y, and Z:
    X = np.arange(MIN, MAX, 0.25)
    Y = np.arange(MIN, MAX, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = (a**2) * (X**2) + (b**2) * (Y**2) - c
    return X, Y, Z

# Define X, Y and Z vectors suitable for a paraboloid.
def plot_paraboloid():
    X, Y, Z =  paraboloid()
    return plot_surface(X, Y, Z)

# Define X, Y and Z vectors suitable for a saddle paraboloid.
def saddle(a = 0.25, b = 0.25, c = 3):
    # Define X, Y, and Z:
    X = np.arange(MIN, MAX, 0.25)
    Y = np.arange(MIN, MAX, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = (a**2) * (X**2) - (b**2) * (Y**2) - c
    return X, Y, Z

# Define X, Y and Z vectors suitable for a saddle paraboloid.
def plot_saddle():
    X, Y, Z =  saddle()
    return plot_surface(X, Y, Z)

# Define X, Y and Z vectors that map an expanding sine wave.
def waves():
    # Make data.
    X = np.arange(MIN, MAX, 0.25)
    Y = np.arange(MIN, MAX, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)
    return X, Y, Z

# Define X, Y and Z vectors that map an expanding sine wave.
def plot_waves():
    X, Y, Z = waves()
    return plot_surface(X, Y, Z)

def plot_surface(X, Y, Z):
    # Define the figure:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(MIN - 0.01, MAX + 0.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    return plt

if __name__ == '__main__':
    print("Here we have a wave, saddle and paraboloid plots:")
    plot_waves().show()
    plot_saddle().show()
    plot_paraboloid().show()
