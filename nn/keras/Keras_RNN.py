#!/usr/bin/env python
# coding: utf-8

# phi - simple LSTM

# Use:
# python Keras_RNN.py non_analogies_csv analogies_csv glove_file
# Or just
# python Keras_RNN.py to run with things in the default locations.

# To do: Figure out where the parameters are set that allows us to
# reproduce the results.

import sys
import csv
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, LSTM


#############################################################
# PARAMETERS - DEFAULTS
#############################################################

default_non_analogies_file = '../.././corpora/verified_non_analogies.csv'
default_analogies_file = '../.././corpora/verified_analogies.csv'
default_glove_directory = '../.././corpora/glove/'
default_glove_file = 'glove.6B.100d.txt'

#############################################################
# Managing command line arguments
#############################################################

num_args = len(sys.argv)

if num_args < 2:
    non_analogies_file = default_non_analogies_file
else:
    non_analogies_file = sys.argv[1]

if num_args < 3:
    analogies_file = default_analogies_file
else:
    analogies_file = sys.argv[2]
    
if num_args < 4:
    glove_file = default_glove_directory + default_glove_file
else:
    glove_file = sys.argv[3]
    
max_sen_length_in_words = 28  # maximum allowed number of words in a sentence
lexicon_size = 10000  # choosing the most 10000 common words # Is this correct?
num_test_samples = 40  # number of testing samples # Consider making this a percent, rather than an absolute number.


#############################################################
# Reading in the CSVs
#############################################################
def readCSV(file_name, sentence_column):
    """
    :param file_name: The file name of the CSV to read
    :param sentence_column: The column within the CSV that contains the sentences.
    Try to make sure this is 1, for consistency.
    :return: A list of the sentences, as strings.
    """
    sent = []
    with open(file_name) as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            sentence = row[sentence_column]
            sent.append(sentence)
    return sent


non_analogy_sentences = readCSV(non_analogies_file, 1)
analogy_sentences = readCSV(analogies_file, 1)

# Create a numpy array of the correct labels.
labels = np.asarray([1] * len(analogy_sentences) + [0] * len(non_analogy_sentences))

# Create a list of all of the sentences
all_sentences = analogy_sentences + non_analogy_sentences

# Tokenize the sentences using the keras tokenizer
# Why are we limiting the number of words to 10000?
tokenizer = Tokenizer(num_words=lexicon_size)
tokenizer.fit_on_texts(all_sentences)
sequences = tokenizer.texts_to_sequences(all_sentences)

word_index = tokenizer.word_index
print("Found {} unique tokens".format(len(word_index)))

# pad sequences to a fixed length
padded_sequences = pad_sequences(sequences, maxlen=max_sen_length_in_words)

# An np array that contains a range of values from 0 to (n-1)
indices = np.arange(padded_sequences.shape[0])

np.random.shuffle(indices)  # shuffle np array that contains the indices

# Arrange the padded sequences and their labels in the same random order.
padded_sequences = padded_sequences[indices]
labels = labels[indices]

# Split into training and testing sets.
train_data = padded_sequences[:len(padded_sequences) - num_test_samples]
test_data = padded_sequences[len(padded_sequences) - num_test_samples:]
train_labels = labels[:len(padded_sequences) - num_test_samples]
test_labels = labels[len(padded_sequences) - num_test_samples:]

print("Shape of data ", padded_sequences.shape)
print("Shape of label", labels.shape)
print("Shape of train_data", train_data.shape)
print("Shape of test_data", test_data.shape)
print("Shape of train_labels", train_labels.shape)
print("Shape of test_labels", test_labels.shape)

embedding_index = {}

# Download glove before this
f = open(glove_file)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coefs
f.close()

print("Found {} words".format(len(embedding_index)))

num_folds = 4
num_val_samples = len(train_data) // num_folds
num_epochs = 10

# Embed sentences
embedding_dim = 100

embedding_matrix = np.zeros((lexicon_size, embedding_dim))
for word, i in word_index.items():
    if i < lexicon_size:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


def build_model():
    model = Sequential()
    model.add(Embedding(lexicon_size, embedding_dim, input_length=max_sen_length_in_words))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    model.save_weights('pre_trained_glove_model.h5')
    return model


# k-fold training
val_acc_history = []
acc_history = []
for i in range(num_folds):
    print('processing fold: #', i)
    val_data = train_data[i * num_val_samples: (i+1)*num_val_samples]
    val_targets = train_labels[i * num_val_samples: (i + 1) * num_val_samples]
    
    partial_train_data = np.concatenate([train_data[:i*num_val_samples],
                                        train_data[(i+1)*num_val_samples:]], axis=0)
    
    partial_test_data = np.concatenate([train_labels[:i * num_val_samples],
                                        train_labels[(i + 1) * num_val_samples:]], axis=0)
    
    model = build_model()
    history = model.fit(partial_train_data, partial_test_data, epochs=num_epochs, batch_size=1, verbose=0,
                        validation_data=(val_data, val_targets))
    val_acc = history.history['val_acc']
    acc = history.history['acc']
    val_acc_history.append(val_acc)
    acc_history.append(acc)


def plot():
    """
    Plots the results.
    :return: No return value.
    """
    avg_acc_history = [np.mean([x[epoch] for x in acc_history]) for epoch in range(num_epochs)]
    avg_val_acc_history = [np.mean([x[epoch] for x in val_acc_history]) for epoch in range(num_epochs)]
    import matplotlib.pyplot as plt
    plt.plot(range(1, len(avg_acc_history) + 1), avg_acc_history, 'bo', label='training acc')
    plt.plot(range(1, len(avg_val_acc_history) + 1), avg_val_acc_history, 'b', label='Validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.show()


results = model.evaluate(test_data, test_labels)
