import numpy as np

def natural_entropy(y):
    return -np.sum(y*np.log(y))

def cross_entropy(y_hat, y):
    return -np.sum(y*np.log(y_hat))

def strange_function():
    """
    What happens when we plot p log q when p = 1-q 
     for: 0 < p <= 1
    """
    x = np.linspace(0, 1, 100, endpoint=False)[1:]
    y = np.flip(x)
    z = np.zeros(x.shape)
    count = 0
    for n in x:
        z[count] = x[count] * np.log(y[count])
        #print("type:", type(n), " value:", n)
        count = count + 1
    return np.array([x,y,z])

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    y_hat = np.array([0.2, 0.3, 0.5])
    y = np.array([0.1, 1, 0.5])
    print("cross entropy: ", cross_entropy(y_hat, y), "   inverse: ", cross_entropy(y, y_hat))
    print("self entropy: ", cross_entropy(y, y), "   inverse: ", cross_entropy(y_hat, y_hat))


    a = strange_function()
    
    print("self entropy: ", cross_entropy(a[0], a[0]))
    plt.scatter(a[0], a[2], alpha=0.5)
    plt.scatter(a[0], a[1], alpha=0.5)
    plt.show()