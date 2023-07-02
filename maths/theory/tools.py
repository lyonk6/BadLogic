import numpy as np

def scatter_vectors(center, sigmas, length, type=np.float32):
    """
        This function returns a numpy array of n dimensional coordinates of type
        numpy.float32. The returned array is of size "(n, length)"
    """
    try:
        dim = len(center)
        if dim != len(sigmas):
            raise ValueError("Parameter 'sigmas' must be the same size as 'center'.")
        arr = np.empty([dim, length], type)

        for i in range(dim):
            arr[i] = np.random.normal(center[i], sigmas[i], length)

        return arr
    except ValueError as err:
        print(err)
    except TypeError as err:
        print(err)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    length = 200
    label1 = np.zeros(length) + 1
    label2 = np.zeros(length) + 2

    sv1 = scatter_vectors([1, 1, 1], [0.4, 0.4, 0.3], length)
    sv2 = scatter_vectors([2, 2, 2], [0.4, 0.6, 0.3], length)

    ## matplotlib chooses two different colors.
    plt.scatter(sv2[0], sv2[1], alpha=0.5, label='x=1; y=1; z=1')
    plt.scatter(sv1[0], sv1[1], alpha=0.5, label='x=2; y=2; z=2')

    ## All one color:
    sv3= np.append(sv1, sv2)
    #plt.scatter(sv3[0], sv3[2], alpha=0.5) 

    print(sv1.shape)
    print(sv2.shape)
    print(sv3.shape)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()