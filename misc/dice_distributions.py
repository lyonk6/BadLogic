import matplotlib.pyplot as plt
import numpy as np
import random

#random.seed(79*37+2)

# Rolls a single die with 'd' sides and return the value
def roll_die(d):
    if d > 100:
        return -1
    else:
        return random.randint(1, d)

# Roll 'n' dice with 'd' sides and return their sum
def roll(n, d):
    sum = 0
    for _ in range(n):
        sum = sum + roll_die(d)
    return sum

# Roll 'n' dice with 'd' sides and return their sum 1000 times.
def roll_distribution(n, d):
    l = []
    for _ in range(1000):
        l.append(roll(n,d))
    return np.array(l, dtype='i')

def plot_hist(rolls):    
    bin = np.arange(start=1, stop=14, step=1, dtype='i')
    print(bin)
    plt.xlim([0, 14])
    plt.hist(rolls, bins=bin, edgecolor='white')
    plt.show()
    plt.cla()


print(roll_die(20))
print(roll_die(3))


if __name__ == '__main__':        
    n2d6  = roll_distribution(2, 6)
    n1d12 = roll_distribution(1, 12)
    n3d4  = roll_distribution(3, 4)
    n4d3  = roll_distribution(4, 3)

    plot_hist(n2d6)
    plot_hist(n1d12)
    plot_hist(n3d4)
    plot_hist(n4d3)