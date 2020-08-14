#!/usr/bin/env python3
""" This script is to do the prediction step of the

text generation TensorFlow tutorial at:
https://www.tensorflow.org/tutorials/text/text_generation

"""

import numpy as np
import tensorflow as tf

from char_rnn import build_model, EMBEDDING_DIM, RNN_UNITS, CHECKPOINT_DIR

# We need an identical dictionary, and a way to map between
# characters and numbers for the embedding.
text_path = "data/pg10900.txt"
with open(text_path, "r") as inputfile:
    # Read the file, replace non-breaking space
    # with a space and remove the byte-order mark.
    # (I think this is windows / latin encoding stuff.
    text = inputfile.read().replace("\xa0", " ").replace("\ufeff", "")

vocab = sorted(set(text))
vocab_size = len(vocab)
char_to_ix = {u:i for i, u in enumerate(vocab)}
ix_to_char = np.array(vocab)


# Prediction time!
# Now we reload the trained weights.
model = build_model(vocab_size, EMBEDDING_DIM, RNN_UNITS, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(CHECKPOINT_DIR))
model.build(tf.TensorShape([1, None]))

print("rebuilt model with a single batch size:")
model.summary()

def generate_text(model, start_string):
    num_generate = 1000

    # convert the input string to numbers
    input_eval = [char_to_ix[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
                
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
