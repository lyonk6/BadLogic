import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class training_data:
    def __init__(self):
        self.index = 0
        self.images = pd.read_csv("data/train.csv")

    def pop_image(self):
        self.index = self.index + 1
        return self.images.iloc[self.index -1]

    def has_next_image(self):
        # How we know the number of rows in a pandas table: df.shape[0]
        if self.index < self.images.shape[0]:
            return True
        else:
            return False


if __name__ == "__main__":
    d = training_data()
    image = d.pop_image()

    print(type(image))
    print(image)

    #Verify we iterate correctly:
    while d.has_next_image():
        if d.index % 5000 == 1:
            print(d.index)
        d.pop_image()
