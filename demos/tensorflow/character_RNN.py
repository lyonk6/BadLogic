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
[encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1 
# Demonstrate how to convert text to and from with the Tokenizer:
print(
    tokenizer.texts_to_sequences(["First"]))
print(
    tokenizer.sequences_to_texts([[20, 2, 8, 3]]))
print(
    type(tokenizer.texts_to_sequences(["First"])))

for c in tokenizer.word_index:
    print("The character '", c, "' has a value of: ", tokenizer.texts_to_sequences([c]))

# Splitting Sequential Data:
train_size = dataset_size * 90 // 100
dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])

#Creating multiple windows.
n_steps = 100
window_length = n_steps + 1 
dataset = dataset.window(window_length, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(window_length))

batch_size = 32
dataset = dataset.shuffle(10000).batch(batch_size)
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
dataset = dataset.map(
    lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch)
)
dataset = dataset.prefetch(1)
