import numpy as np
import tools
def natural_entropy(y):
    return -np.sum(y*np.log(y))

def cross_entropy(y_hat, y):
    return -np.sum(y*np.log(y_hat))

def conditional_entropy(x, y_given_x):
    """
    One applications of conditional entropy in cryptography is evaluating
    the strength of encryption keys.  The conditional entropy can be used
    to quantify the amount of uncertainty that remains about the 
    encryption key, given knowledge of the corresponding ciphertext.
    """
    return -np.sum(y_given_x * (np.log(y_given_x) / np.log(x)))


def strange_function():
    """
    What happens when we plot p log q when p = 1-q 
     for: 0 < p <= 1
    """
    # X: [0.01 0.02 0.03 0.04 0.05 0.06 ...
    # Y: [0.99 0.98 0.97 0.96 0.95 0.94 0.93 ...
    # Z: [-1.01e-04 -4.04e-04 -9.14e-04 -1.63e-03 ...
    x = np.linspace(0, 1, 100, endpoint=False)[1:]
    y = np.flip(x)
    z = np.zeros(x.shape)
    z = x * np.log(y)
    return np.array([x,y,z])

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    a = strange_function()

    plt.scatter(a[0], a[2], alpha=0.5, label='p * log(q)')
    plt.scatter(a[0], a[1], alpha=0.5, label='q = 1 - p')
    plt.xlabel('p')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    length = 200
    sv1 = tools.scatter_vectors([1, 1, 1], [0.4, 0.4, 0.3], length)
    sv2 = tools.scatter_vectors([2, 2, 2], [0.4, 0.6, 0.3], length)
    print("cross entropy: ", cross_entropy(a[0], a[0]))
