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
    z = -x * np.log(y)
    return np.array([x,y,z])

def softmax(x):
    return(np.exp(x)/np.exp(x).sum())

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    a = strange_function()

    fig, axs = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')
    #axs[0].plot(x, y)
    #axs[1].plot(x, -y)

    axs[0].plot(a[0], a[2], alpha=0.5, label='-p * log(1-p)')
    axs[0].plot(a[0], a[1], alpha=0.5, label='1 - p')
    axs[0].set_xlabel('p')
    axs[0].set_ylabel('H(p,q)')
    axs[0].legend()
    
    axs[1].plot(a[0], -a[0]*np.log(a[0]), alpha=0.5, label='-p * log(p)')
    axs[1].set_xlabel('p')
    axs[1].set_ylabel('S')
    axs[1].legend()


    #plt.legend()
    plt.show()
    length = 200
    sv1 = tools.scatter_vectors([1, 1, 1], [0.5, 0.5, 0.5], length)
    sv2 = tools.scatter_vectors([1, 2, 5], [0.5, 0.5, 0.5], length)
    print("Natural Entropy X: ", natural_entropy(softmax(sv2[0])))
    print("Natural Entropy Y: ", natural_entropy(softmax(sv2[1])))
    print("Natural Entropy Z: ", natural_entropy(softmax(sv2[2])))


    print("Cross Entropy X sv1,sv2 ", cross_entropy(softmax(sv1[0]), softmax(sv2[0])))
    print("Cross Entropy Y sv1,sv2 ", cross_entropy(softmax(sv1[1]), softmax(sv2[1])))
    print("Cross Entropy Z sv1,sv2 ", cross_entropy(softmax(sv1[2]), softmax(sv2[2])))

    print("Cross Entropy X sv2,sv1 ", cross_entropy(softmax(sv2[0]), softmax(sv1[0])))
    print("Cross Entropy Y sv2,sv1 ", cross_entropy(softmax(sv2[1]), softmax(sv1[1])))
    print("Cross Entropy Z sv2,sv1 ", cross_entropy(softmax(sv2[2]), softmax(sv1[2])))

    ## matplotlib chooses two different colors.
