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

    sv1 = scatter_vectors([1, 1, 1], [1.4, 0.9, 0.8], length)
    sv2 = scatter_vectors([1, 3, 5], [0.8, 0.6, 1.3], length)

    ## matplotlib chooses two different colors.

    fig, axs = plt.subplots(1,3)
    #axs[0].plot(x, y)
    #axs[1].plot(x, -y)

    axs[0].scatter(sv2[0], sv2[1], alpha=0.5, label='x=1; y=1; ')
    axs[0].scatter(sv1[0], sv1[1], alpha=0.5, label='x=2; y=2; ')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].legend()

    axs[1].scatter(sv2[0], sv2[2], alpha=0.5, label='x=1; z=1; ')
    axs[1].scatter(sv1[0], sv1[2], alpha=0.5, label='x=2; z=2; ')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    axs[1].legend()

    axs[2].scatter(sv2[1], sv2[2], alpha=0.5, label='y=1; z=1; ')
    axs[2].scatter(sv1[1], sv1[2], alpha=0.5, label='y=2; z=2; ')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('y')
    axs[2].legend()

    ## All one color:
    sv3= np.append(sv1, sv2)
    #plt.scatter(sv3[0], sv3[2], alpha=0.5) 

    print(sv1.shape)
    print(sv2.shape)
    print(sv3.shape)
    plt.show()