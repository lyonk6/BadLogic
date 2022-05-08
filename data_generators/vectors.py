import numpy as np

class ScatterVectors:
    def __init__(self, center, sigmas, length, type=np.float32):
        try:
            self.dim   = len(center)
            if self.dim != len(sigmas):
                raise ValueError("Parameter 'sigmas' must be the same size as 'center'.")
            self.array = np.empty([self.dim, length], type)

            for i in range(self.dim):
                self.array[i] = np.random.normal(center[i], sigmas[i], length)

        except ValueError as err:
            print(err)
        except TypeError as err:
            print(err)

    def getArray(self):
        return self.array

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    length = 1000
    sv = ScatterVectors([0, 0, 0], [0.1, 0.2, 0.3], length)

    plt.scatter(sv.getArray()[0],sv.getArray()[2], alpha=0.5)
    plt.show()