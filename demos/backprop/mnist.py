import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class training_data:
    def __init__(self):
        self.index = 0
        self.images = pd.read_csv("data/train.csv")

    def pop_image_and_label(self):
        self.index = self.index + 1
        data = self.images.iloc[self.index -1].to_numpy()
        # X, label
        return np.array([data[1:]]).T / 255., data[:1]

    def has_next_image(self):
        # The number of rows in a pandas table is given by: df.shape[0]
        if self.index < self.images.shape[0]:
            return True
        else:
            return False

def plot_image_v2(image):
    image = image.reshape((28,28))

    # plot the sample
    fig = plt.figure
    plt.imshow(image, cmap='gray')
    plt.show()


if __name__ == "__main__":
    d = training_data()
    d.pop_image_and_label()
    d.pop_image_and_label()

    image, label = d.pop_image_and_label()
    plot_image_v2(image)
    print("This is the label: ", label[0])

    #Verify we iterate correctly:
    while d.has_next_image():
        if d.index % 5000 == 1:
            print(d.index)
        d.pop_image_and_label()
