#!/usr/bin/env python3
""" This is a script to run through the tensorflow text tutorial
    see: https://www.tensorflow.org/tutorials/text/text_generation
    This code is mostly copied from there.
"""

import os

import numpy as np
import tensorflow as tf
# Use the alternative regex module
# as it includes punctuation
import regex as re


EPOCHS = 10
BATCH_SIZE = 32
BUFFER_SIZE = 500
EMBEDDING_DIM = 64
RNN_UNITS = 128
CHECKPOINT_DIR = "./training_checkpoints"
#TEXT_PATH = "./data/star_wars.txt"
TEXT_PATH = "./data/pg10900.txt"


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def split_input_target(chunk):
    """ Each sequence needs to be split into an input and a target text """
    input_text = chunk[:-1]
    target_text = chunk[1:]
    
    return input_text, target_text

def get_vocab():
    """ Load up the text file, pre-process and extract vocab. """
    
    with open(TEXT_PATH, "r") as inputfile:
        # Read the file, replace non-breaking space
        # with a space and remove the byte-order mark.
        # (I think this is windows / latin encoding stuff.
        text = inputfile.read().replace("\xa0", " ").replace("\ufeff", "")

    # Check the characters have been correctly loaded up:
    print("unique characters are:\n", sorted(set(text)))

    print(text[10000:10050])
    print("input text contains {} characters".format(len(text)))

    # Now - pre-process by replacing any repeated spaces with single spaces.
    text = re.sub(r' +', ' ', text)

    # Deal with numbers
    text = re.sub(r'\d+', '(number)', text)

    # Then we want to split into individual words.
    # Options:
    #    * Regulate cases?

    # I want to split on whitespace or punctiation.
    text_split = re.split(r"([\W+\p])", text)
    print("input text contains {} words".format(len(text_split)))

    # The plan is to use a small-dimension embedding +
    # a LSTM/GRU approach. I don't think I can code up
    # a transformer / attention model on this hardware.

    # Following the tutorial - extract the number of unique characters.
    # (letters, punctuation, weird bytes)
    vocab = set(text_split)
    print("vocab contains {} words".format(len(vocab)))
    
    # First step in the processing is transforming the
    # massive string into an array of integers.
    word_to_ix = {u:i for i, u in enumerate(vocab)}
    ix_to_word = np.array(list(vocab))

    return vocab, text_split, word_to_ix, ix_to_word
    

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    """ Build the recurrent neural network """
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


if __name__ == "__main__":

    # Load up the text file
    vocab, text, char_to_ix, ix_to_char  = get_vocab()

    # This is a numpy array of 32bit integers representing
    # the complete text.
    text_as_int = np.array([char_to_ix[char] for char in text])

    # Now we need to split the array into a sequences of a given length
    # Take in 100 characters, predict the next one.
    seq_length = 30
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
        
    # Pack this into batches
    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    # Next step: we have data, now we build the model
    # The tutorial uses a GRU, so why not?
    vocab_size = len(vocab)

    model = build_model(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        rnn_units=RNN_UNITS,
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
    print("Input: \n", "".join(ix_to_char[input_example_batch[0].numpy()]))
    print()
    print("Next char predictions: \n", "".join(ix_to_char[sampled_indices]))

    # Now: Train the model!
    ######################
    example_batch_loss = loss(target_example_batch, example_batch_predictions)
    print("prediction shape: ", example_batch_predictions.shape)
    print("scalar loss: ", example_batch_loss.numpy().mean())

    model.compile(optimizer="adam", loss=loss)

    # Set up callbacks to save out the model state
    checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
