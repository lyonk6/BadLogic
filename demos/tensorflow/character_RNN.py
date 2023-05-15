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
print("This is the type of the encoded input:", type(encoded))
print("This is shape of the encoded input:", encoded.shape)
# Demonstrate how to convert text to and from with the Tokenizer:

"""
for c in tokenizer.word_index:
    print("The character '", c, "' has a value of: ", tokenizer.texts_to_sequences([c])[0][0])

"""


print(shakespeare_text[:148])


"""
model = keras.models.Sequential([
    keras.layers.GRU(128, return_sequences=True, stateful=True, dropout=0.2, recurrent_dropout=0.2, batch_input_shape=[batch_size, None, max_id]),
    keras.layers.GRU(128, return_sequences=True, stateful=True, dropout=0.2, recurrent_dropout=0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(max_id, activation="softmax"))
])

class ResetStatesCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs):
        self.model.reset_states()

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
model.fit(dataset, epochs=50, callbacks=[ResetStatesCallback()])
"""