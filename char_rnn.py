#!/usr/bin/env python3
""" This is a script to run through the tensorflow text tutorial
    see: https://www.tensorflow.org/tutorials/text/text_generation
    This code is mostly copied from there.
"""

import numpy as np


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

