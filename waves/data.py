import numpy as np

MIN = 10.0
MAX = 10.0

# Let a, b, c be constants > 0 for the function:
#  Z = (A**2) * (X**2) + (B**2) * (Y**2) - c
def sample_paraboloid(a = 0.25, b = 0.25, c = 5):
    x = random.uniform(MIN, MAX)
    y = random.uniform(MIN, MAX)
    z = (a**2) * (x**2) + (b**2) * (y**2) - c
    return x, y, z

# Let a, b, c be constants > 0 for the function:
#  Z = (A**2) * (X**2) - (B**2) * (Y**2) - c
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




if __name__ == '__main__':
    print("data ")
