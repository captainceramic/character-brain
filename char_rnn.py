#!/usr/bin/env python3
""" This is a script to run through the tensorflow text tutorial
    see: https://www.tensorflow.org/tutorials/text/text_generation
    This code is mostly copied from there.
"""

import os
import numpy as np

import tensorflow as tf

EPOCHS = 5
BATCH_SIZE = 16
BUFFER_SIZE = 2500

# First step: load up the text
text_path = "data/pg10900.txt"
with open(text_path, "r") as inputfile:
    # Read the file, replace non-breaking space
    # with a space and remove the byte-order mark.
    # (I think this is windows / latin encoding stuff.
    text = inputfile.read().replace("\xa0", " ").replace("\ufeff", "")

print(text[10000:10100])
print("input text contains {} characters".format(len(text)))

# The plan is to use a small-dimension embedding +
# a LSTM/GRU approach. I don't think I can code up
# a transformer / attention model on this hardware.

# Following the tutorial - extract the number of unique characters.
# (letters, punctuation, weird bytes)
vocab = sorted(set(text))

# First step in the processing is transforming the
# massive string into an array of integers.
char_to_ix = {u:i for i, u in enumerate(vocab)}
ix_to_char = np.array(vocab)

# This is a numpy array of 32bit integers representing
# the complete text.
text_as_int = np.array([char_to_ix[char] for char in text])

# Now we need to split the array into a sequences of a given length
# Take in 100 characters, predict the next one.
seq_length = 100
examples_per_epoch = len(text) // (seq_length + 1)

# Use the tensorflow dataset
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
# have a look at some letters:
for i in char_dataset.take(5):
    print(ix_to_char[i.numpy()])

# Now we batch these up
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
for item in sequences.take(5):
    print(''.join(ix_to_char[item.numpy()]))

# Each sequence needs to be split into an input and a target text
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]

    return input_text, target_text

# Pack this into batches
dataset = sequences.map(split_input_target)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Next step: we have data, now we build the model
# The tutorial uses a GRU, so why not?
vocab_size = len(vocab)
embedding_dim = 64
rnn_units = 128

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])

    return model

model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

# check the input batches and the model shape
for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape)

model.summary()

# The output will be logits, we sample from there
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

# decode the sampled indices to look at some raw output on a batch
print("Input: \n", "".join(ix_to_char[input_example_batch[0]]))
print()
print("Next char predictions: \n", "".join(ix_to_char[sampled_indices]))

# Now: Train the model!
######################
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss = loss(target_example_batch, example_batch_predictions)
print("prediction shape: ", example_batch_predictions.shape)
print("scalar loss: ", example_batch_loss.numpy().mean())

model.compile(optimizer="adam", loss=loss)

# Set up callbacks to save out the model state
checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

# Prediction time!
# Now we reload the trained weights.
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

print("rebuilt model with a single batch size:")
model.summary()

def generate_text(model, start_string):
    num_generate = 1000

    # convert the input string to numbers
    input_eval = [char_to_ix[s] for s in start_string]
                
    text_generated = []

    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature

        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(ix_to_char[predicted_id])

    return start_string + ''.join(text_generated)

print(generate_text(model, start_string="And then God said"))
