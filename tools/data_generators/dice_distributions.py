import matplotlib.pyplot as plt
import numpy as np
import random

random.seed(79*37+2)

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

# Roll 'n' dice with 'd' sides and return the sum of 's' samples.
def roll_distribution(n, d, s=1000):
    print("n dice: ", n, "   sides:", d, "  samples:", s)
    l = []
    for _ in range(s):
        l.append(roll(n,d))
    return np.array(l, dtype='i')

def plot_hist(rolls):
    min_val = np.min(rolls)
    max_val = np.max(rolls)
    bin = np.arange(start=min_val - 1 , stop=max_val + 1, step=1, dtype='i')
    plt.xlim([0, max_val+1])
    plt.hist(rolls, bins=bin, edgecolor='white')
    plt.show()
    plt.cla()



if __name__ == '__main__':        
    plot_hist(roll_distribution(1, 6))
    plot_hist(roll_distribution(2, 6))
    plot_hist(roll_distribution(3, 6))
    plot_hist(roll_distribution(4, 6))
    plot_hist(roll_distribution(5, 6))
    plot_hist(roll_distribution(6, 6))

    plot_hist(roll_distribution(4, 10, 1))
    plot_hist(roll_distribution(4, 10, 2))
    plot_hist(roll_distribution(4, 10, 3))
    plot_hist(roll_distribution(4, 10, 4))
    plot_hist(roll_distribution(4, 10, 5))

    plot_hist(roll_distribution(4, 10, 10))
    plot_hist(roll_distribution(4, 10, 20))
    plot_hist(roll_distribution(4, 10, 30))
    plot_hist(roll_distribution(4, 10, 40))
    plot_hist(roll_distribution(4, 10, 50))
    plot_hist(roll_distribution(4, 10, 60))
    plot_hist(roll_distribution(4, 10, 70))
    plot_hist(roll_distribution(4, 10, 80))


    plot_hist(roll_distribution(4, 8, 100))
    plot_hist(roll_distribution(4, 8, 200))
    plot_hist(roll_distribution(4, 8, 300))
    plot_hist(roll_distribution(4, 8, 400))
    plot_hist(roll_distribution(4, 8, 500))
    plot_hist(roll_distribution(4, 8, 600))
    plot_hist(roll_distribution(4, 8, 700))
    plot_hist(roll_distribution(4, 8, 800))
    plot_hist(roll_distribution(4, 8, 900))
    plot_hist(roll_distribution(4, 8, 1000))