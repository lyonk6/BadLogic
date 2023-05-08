import numpy as np
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
print(keras.__version__)


(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data()
print("Shape of x_train",  x_train.shape)
print("Shape of y_train",  y_train.shape)
print("Shape of x_test",  x_test.shape)
print("Shape of y_test",  y_test.shape)

print(x_train)
print(x_train[0][:10])

word_index = keras.datasets.imdb.get_word_index()
id_to_word = {id_ + 3: word for word, id_ in word_index.items()}
for id_, token in enumerate(("<pad>", "<sos>", "<unk>")):
    id_to_word[id_] = token

" ".join([id_to_word[id_] for id_ in x_train[0][:10]])


import tensorflow_datasets as tfds

datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
train_size = info.splits["train"].num_examples

def preprocess(x_batch, y_batch):
    x_batch = tf.strings.substr(x_batch, 0, 300)
    x_batch = tf.strings.regex_replace(x_batch, b"<br\\s*/?>", b" ")
    x_batch = tf.strings.regex_replace(x_batch, b"[^a-zA-Z']", b" ")
    x_batch = tf.strings.split(x_batch)
    return x_batch.to_tensor(default_value=b""), y_batch

from collections import Counter
vocabulary = Counter()
for x_batch, y_batch in datasets["train"].batch(32).map(preprocess):
    for review in x_batch:
        vocabulary.update(list(review.numpy()))

vocabulary.most_common()[:3]

vocab_size = 10000
truncated_vocabulary = [
    word for word, count in vocabulary.most_common()[:vocab_size]
]

words = tf.constant(truncated_vocabulary)
words_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
vocab_init = tf.lookup.KeyValueTensorInitializer(words, words_ids)
num_oov_buckets = 1000
table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)

print(table.lookup(tf.constant([b"This movie was faaaaaantastic".split()])))

def encode_words(x_batch, y_batch):
    return table.lookup(x_batch), y_batch

train_set = datasets["train"].batch(32).map(preprocess)
train_set = train_set.map(encode_words).prefetch(1)


embed_size = 128
model = keras.models.Sequential([
            keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size,
                input_shape=[None]),
            keras.layers.GRU(128, return_sequneces=True),
            keras.layers.GRU(128),
            keras.layers.Dense(1, activation="sigmoid")
        ])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(train_set, epochs=5)