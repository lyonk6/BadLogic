import matplotlib.pyplot as plt
import numpy as np
import random

def roll_die(d):
    # Rolls a single die with 'd' sides and return the value.
    if d > 100:
        return -1
    else:
        return random.randint(1, d)

def roll(n, d):
    #Roll 'n' dice with 'd' sides and return their sum.
    sum = 0
    for _ in range(n):
        sum = sum + roll_die(d)
    return sum

def roll_distribution(n, d, s=1000):
    """
    Roll 'n' dice with 'd' sides and return the sums of 's' samples.
         n dice:  50    sides: 2   samples: 1000
        [69, 75, 69, 79, 77, 74, 67, 76, 77, 73, 76, 73, 73, 78,  ...
    """
    # print("n dice: ", n, "   sides:", d, "  samples:", s)
    l = []
    for _ in range(s):
        l.append(roll(n,d))
    return np.array(l, dtype='i')

def plot_hist(rolls):
    """
    # print(rolls)
    #    [69, 75, 69, 79, 77, 74, 67, 76, 77, 73, 76, 73, 73, 78,  ...
    """
    min_val = np.min(rolls)
    max_val = np.max(rolls)
    bin = np.arange(start=min_val - 1 , stop=max_val + 2, step=1, dtype='i')
    # Include x=0 to illustrate how the mean changes. 
    plt.xlim([0, max_val+2])
    plt.hist(rolls, bins=bin, edgecolor='white')
    plt.show()
    plt.cla()

def theoretical_distribution():
    a = np.zeros((6,6,6))
    for i in range(0,6):
        for j in range(0,6):
            for k in range(0,6):
                a[i][j][k]=3+i+j+k
    return np.reshape(a, (1,-1))


if __name__ == '__main__':
    random.seed(42)
    plot_hist(roll_distribution(1, 6))
    plot_hist(roll_distribution(2, 6))
    plot_hist(roll_distribution(3, 6))
    plot_hist(roll_distribution(5, 6))
    plot_hist(roll_distribution(8, 6))
    plot_hist(roll_distribution(13, 6))
    plot_hist(roll_distribution(21, 6))


    plot_hist(roll_distribution(1, 2))
    plot_hist(roll_distribution(10, 2))
    plot_hist(roll_distribution(20, 2))
    plot_hist(roll_distribution(30, 2))
    plot_hist(roll_distribution(40, 2))
    plot_hist(roll_distribution(50, 2))
    a  = theoretical_distribution()
    print(plot_hist(a[0]))
    print(a)
