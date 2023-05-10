import numpy as np
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
print(keras.__version__)


#First fetch data for out character RNN:
shakespeare_url = "https://homl.info/shakespeare"
filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)

with open(filepath) as f:
    shakespeare_text = f.read()
    # Print the entire play
    #print(shakespeare_text)


# The keras tokenizer converts text to numeric tokens. In this
# model we are encoding each character as a token so char_level
# is set to 'True'
tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts([shakespeare_text])
max_id = len(tokenizer.word_index) # count distinct chars
dataset_size = tokenizer.document_count # total chars

# list: [a], ndarray: a
#encoded = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1
encoded = np.array(tokenizer.texts_to_sequences([list(shakespeare_text)])) - 1
print("This is the time of the encoded input:", type(encoded))
print("This is shape of the encoded input:", encoded.shape)
# Demonstrate how to convert text to and from with the Tokenizer:
print(
    tokenizer.texts_to_sequences(["First"]))
print(
    tokenizer.sequences_to_texts([[20, 2, 8, 3]]))
print(
    type(tokenizer.texts_to_sequences(["First"])))

for c in tokenizer.word_index:
    print("The character '", c, "' has a value of: ", tokenizer.texts_to_sequences([c])[0][0])



# Splitting Sequential Data:
train_size = dataset_size * 90 // 100
print("This is how big our training size is: ", train_size)
dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])
print("Dataset type: ", type(dataset))
#Creating multiple windows.
n_steps = 100
window_length = n_steps + 1 
dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])
dataset = dataset.window(window_length, shift=n_steps, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(window_length))
dataset = dataset.batch(1)
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
dataset = dataset.map(lambda x_batch, y_batch: (tf.one_hot(x_batch, depth=max_id), y_batch))
dataset = dataset.prefetch(1)


