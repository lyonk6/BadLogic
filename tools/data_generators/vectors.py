import numpy as np

# Here is a function that creates vectors:

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




# And here is a Class that does the same thing:
class ScatterVectors:
    def __init__(self, center, sigmas, length, type=np.float32):
        try:
            self.dim   = len(center)
            if self.dim != len(sigmas):
                raise ValueError("Parameter 'sigmas' must be the same size as 'center'.")
            self.arr = np.empty([self.dim, length], type)

            for i in range(self.dim):
                self.arr[i] = np.random.normal(center[i], sigmas[i], length)

        except ValueError as err:
            print(err)
        except TypeError as err:
            print(err)

    def array(self):
        return self.arr

    def append(self, other):
        try:
            print(type(self.arr.dtype))
            print(type(other.arr.dtype))
            if self.dim != other.dim:
                raise ValueError("Error. ScatterVectors have incompatable dimensions.")
            self.arr = np.append(self.arr, other.arr, axis=1)
            return self
        except ValueError as err:
            print(err)
        except TypeError as err:
            print(err)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    length = 200
    label1 = np.zeros(length) + 1
    label2 = np.zeros(length) + 2

    sv1 = ScatterVectors([1, 1, 1], [0.4, 0.4, 0.3], length)
    sv2 = ScatterVectors([2, 2, 2], [0.4, 0.6, 0.3], length)

    ## matplotlib chooses two different colors.
    plt.scatter(sv2.array()[0], sv2.array()[1], alpha=0.5)
    plt.scatter(sv1.array()[0], sv1.array()[1], alpha=0.5)

    ## All one color:
    sv3= sv2.append(sv1)
    #plt.scatter(sv3.array()[0], sv3.array()[2], alpha=0.5) 

    print(sv1.array().shape)
    print(sv2.array().shape)
    print(sv3.array().shape)
    print(sv3.array().dtype)
    plt.show()