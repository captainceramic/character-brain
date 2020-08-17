#!/usr/bin/env python3
""" This script is to do the prediction step of the

text generation TensorFlow tutorial at:
https://www.tensorflow.org/tutorials/text/text_generation

"""

import numpy as np
import tensorflow as tf

from char_rnn import EMBEDDING_DIM, RNN_UNITS, CHECKPOINT_DIR
from char_rnn import build_model, get_vocab


# We need an identical dictionary, and a way to map between
# characters and numbers for the embedding.
vocab, text, char_to_ix, ix_to_char  = get_vocab()
vocab_size = len(vocab)

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

    print(text_generated)
        
    return ''.join(start_string + text_generated)

print(generate_text(model, start_string="and then God said".split()))
