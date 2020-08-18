#!/usr/bin/env python3
""" This is a script to run through the tensorflow text tutorial
    see: https://www.tensorflow.org/tutorials/text/text_generation
    This code is mostly copied from there.
"""

import os
import re

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

EPOCHS = 10
BATCH_SIZE = 32
BUFFER_SIZE = 500
EMBEDDING_DIM = 64
RNN_UNITS = 128
CHECKPOINT_DIR = "./training_checkpoints"
#TEXT_PATH = "./data/star_wars.txt"
TEXT_PATH = "./data/pg10900.txt"
VOCAB_PATH = "./data/vocab_encoder.dat"


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def split_input_target(chunk):
    """ Each sequence needs to be split into an input and a target text """
    input_text = chunk[:-1]
    target_text = chunk[1:]
    
    return input_text, target_text

def get_vocab():
    """ Load up the text file, pre-process and extract vocab.

    Rather than do this manually, I use the TensorFlow datasets
    subword encoder.

    """
    
    with open(TEXT_PATH, "r") as inputfile:
        # Read the file, replace non-breaking space
        # with a space and remove the byte-order mark.
        # (I think this is windows / latin encoding stuff.
        text = inputfile.read().replace("\xa0", " ").replace("\ufeff", "")

    try:
        encoder = tfds.features.text.SubwordTextEncoder.load_from_file(VOCAB_PATH)
    except tf.errors.NotFoundError as err:
        # Check the characters have been correctly loaded up:
        print("unique characters are:\n", sorted(set(text)))

        # Now use the subword encoder
        split_text = re.split(r"(\n)", text)
        print("input text contains {} lines".format(len(split_text)))
        
        encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            split_text, target_vocab_size=2**13)
        encoder.save_to_file(VOCAB_PATH)
        
    return encoder, text


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
    encoder, raw_text  = get_vocab()

    # This is a numpy array of 32bit integers representing
    # the complete text.
    text_as_int = encoder.encode(raw_text)

    # Now we need to split the array into a sequences of a given length
    # Take in X subwords, predict the next one.
    seq_length = 40
    examples_per_epoch = len(text_as_int) // (seq_length + 1)

    # Use the tensorflow dataset
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    # have a look at some letters:
    for i in char_dataset.take(5):
        print(encoder.decode([i.numpy()]))

    # Now we batch these up
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
    for item in sequences.take(5):
        print(''.join(encoder.decode(item.numpy())))
        
    # Pack this into batches
    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    # Next step: we have data, now we build the model
    # The tutorial uses a GRU, so why not?

    vocab_size = encoder.vocab_size
    model = build_model(
        vocab_size=vocab_size,
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
    print("Input: \n", "".join(encoder.decode(input_example_batch[0].numpy())))
    print("Next char predictions: \n", "".join(encoder.decode(sampled_indices)))

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
