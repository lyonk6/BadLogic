import numpy as np

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
            print("check!")
            # todo merge two arrays:
            self.arr = np.append(self.arr, other.arr, axis=1)
            return self
        except ValueError as err:
            print(err)
        except TypeError as err:
            print(err)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    length = 10
    sv = ScatterVectors([0, 0, 0], [0.1, 0.2, 0.3], length)
    sv2= ScatterVectors([2, 2, 2], [0.3, 0.6, 0.1], length)
    sv3= sv2.append(sv)
    plt.scatter(sv3.array()[0],sv3.array()[2], alpha=0.5)
    print(sv.array().shape)
    print(sv2.array().shape)
    print(sv3.array().shape)
    print(sv3.array().dtype)
    plt.show()