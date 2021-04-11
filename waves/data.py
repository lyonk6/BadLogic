import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

MIN = -10
MAX = 10
IMAGE_WIDTH = 256

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
def plot_paraboloid(a = 0.25, b = 0.25, c = 5):
    # Define X, Y, and Z:
    X = np.arange(MIN, MAX, 0.25)
    Y = np.arange(MIN, MAX, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = (a**2) * (X**2) + (b**2) * (Y**2) - c
    return plot_surface(X, Y, Z)

# Define X, Y and Z vectors suitable for a saddle paraboloid.
def plot_saddle(a = 0.25, b = 0.25, c = 3):
    # Define X, Y, and Z:
    X = np.arange(MIN, MAX, 0.25)
    Y = np.arange(MIN, MAX, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = (a**2) * (X**2) - (b**2) * (Y**2) - c
    return plot_surface(X, Y, Z)

# Define X, Y and Z vectors that map an expanding sine wave.
def plot_waves():
    # Make data.
    X = np.arange(MIN, MAX, 0.25)
    Y = np.arange(MIN, MAX, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)
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


def sample_to_one_hot(x):
    xT = torch.zeros((1, IMAGE_WIDTH), dtype=torch.float64)
    adjustment = (IMAGE_WIDTH-1)/(MAX - MIN)
    position = int(np.floor((adjustment * (x + 6)) + 0.5).item())
    #print("Here is our sample x value and it's index: ", x, ", ", position)
    xT[0][position]=1
    return xT

if __name__ == '__main__':
    print("Here we have a wave, saddle and paraboloid plots:")
    plot_waves().show()
    plot_saddle().show()
    plot_paraboloid().show()
