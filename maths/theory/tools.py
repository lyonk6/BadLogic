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
