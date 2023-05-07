import tensorflow as tf
from tensorflow import keras


# Load fashion mnist data from Keras. 
# Returns: X_valid, X_train, y_valid, y_train, class_names
def load_data_with_keras():
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    #print("Here is the full input set: ", X_train_full)
    print("Here is the full input data-type: ", X_train_full.dtype)
    print("Here is the full input object type: ", type(X_train_full))
    print("Here are the y values: ", y_train_full)
    X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    # We need to convert digital labels (0-9) to human readable ones:
    class_names = ["T-Shirt", "Pants", "Pull-Over", "Dress", "Coat",
        "Sandal", "Shirt", "Sneakers", "Bag", "Ankle Boot"]
    return X_valid, X_train, y_valid, y_train, class_names
    
def make_model():
    model = keras.models.Sequential()
    ## TODO what does "keras.layers.Flatten" do?
    model.add(keras.layers.Flatten(input_shape=[28,28]))
    model.add(keras.layers.Dense (300, activation="relu"))
    model.add(keras.layers.Dense (100, activation="relu"))
    model.add(keras.layers.Dense (10, activation="softmax"))

if __name__ == "__main__":
    print("TensorFlow Version: ", tf.__version__)
    print("Keras Version: ", keras.__version__)
    load_data_with_keras()
    print("Done")
