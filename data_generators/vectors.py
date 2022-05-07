import numpy as np

class ScatterVectors:
    def __init__(self, type, dim, samples, centers):
    try:
        #Arrays should be constructed using array, zeros or empty
        #numpy.empty(shape, dtype=float, order='C', *, like=None)
        self.center = center

        break
    except ValueError:
        print("Oops!  That was no valid number.  Try again")

    def scatterPoints(self, samples,  ):

if __name__ == "__main__":
    # Creating normally distributed results:
    mu, sigma, length = 0, 0.2, 1000 # mean, standard deviation, and size
    s = np.random.normal(mu, sigma, length)

    # Plot a single vector and see if it's normally distributed:
    import matplotlib.pyplot as plt
    count, bins, ignored = plt.hist(s, 30, density=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
    plt.show()

    # Plot a second vector and create a scatter plot:
    t = np.random.normal(mu + 0.5, sigma*2, length)
    plt.scatter(s, t, alpha=0.3)

    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.show()


