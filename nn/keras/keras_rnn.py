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
import time
from datetime import datetime
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras_rnn_utils import readCSV, produce_embedding_index

#############################################################
# PARAMETERS - DEFAULTS
#############################################################
from keras_rnn_utils import evaluate_model, train_model

default_non_analogies_file = '../../corpora/verified_non_analogies.csv'
default_analogies_file = '../../corpora/verified_analogies.csv'
default_glove_directory = '../../corpora/glove/'
default_glove_file = 'glove.6B.100d.txt'
default_results_file = './results/results-variability-test.txt'

num_folds = 4
num_epochs = 10

# Embed sentences # Is this the number of dimensions of the embedding vectors?
embedding_dim = 100

max_sen_length_in_words = 28  # maximum allowed number of words in a sentence
lexicon_size = 10000  # choosing the most 10000 common words # Is this correct?
num_test_samples = 40  # number of testing samples # Consider making this a percent, rather than an absolute number.
epoch_batch_size = 1  # Size of the batch in each epoch.

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

# Start the timer
start_time = time.time()
start_datetime = datetime.now()

analogy_sentences = readCSV(analogies_file, 1)
num_analogy_sentences = len(analogy_sentences)
non_analogy_sentences = readCSV(non_analogies_file, 1)
num_non_analogy_sentences = len(non_analogy_sentences)

# Create a numpy array of the correct labels.
labels = np.asarray([1] * num_analogy_sentences + [0] * num_non_analogy_sentences)

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
num_val_samples = len(train_data) // num_folds

embedding_index = produce_embedding_index(glove_file)

embedding_matrix = np.zeros((lexicon_size, embedding_dim))
for word, i in word_index.items():
    if i < lexicon_size:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


# k-fold training - Each fold chooses a new slice of the training data to be the validation set
# This should probably be its own function - likely a mutator that takes the model as an argument?
model, val_acc_histories, acc_histories = train_model(num_val_samples, train_data, train_labels, lexicon_size,
                                                      embedding_dim, max_sen_length_in_words, embedding_matrix,
                                                      num_epochs, epoch_batch_size, num_folds)

test_loss, test_accuracy, results_confusion_matrix = evaluate_model(model, test_data, test_labels)

# Stop the timer
stop_time = time.time()
elapsed_time = stop_time - start_time


# This will need to record all of the parameters used, along with the results.
def record_results(results_file=default_results_file):
    print("Recording results...")
    results_handler = open(results_file, 'a')
    results_handler.write("="*30 + "\n")
    results_handler.write("Start time: " + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + "\n")
    results_handler.write("Elapsed time (s): " + str(elapsed_time) + "\n")

    # Record information about the training data
    results_handler.write("Analogies file: " + analogies_file + "\n")
    results_handler.write("Non-Analogies file: " + non_analogies_file + "\n")
    results_handler.write("Analogies: " + str(num_analogy_sentences) +
                          " Non-Analogies: " + str(num_non_analogy_sentences) +
                          " Total: " + str(num_analogy_sentences + num_non_analogy_sentences) + "\n")

    # Record information about the NN setup
    results_handler.write("Embedding vector dimensions: " + str(embedding_dim) + "\n")

    # Record information about the training.
    results_handler.write("Training epochs: " + str(num_epochs) + "\n")
    results_handler.write("Epoch batch size: " + str(epoch_batch_size) + "\n")
    results_handler.write("Training folds: " + str(num_folds) + "\n")

    # Record information about the results.
    results_handler.write("Testing Loss: " + str(test_loss) + "\n")
    results_handler.write("Testing Accuracy: " + str(test_accuracy) + "\n")
    results_handler.write("Confusion Matrix:\n " + str(results_confusion_matrix) + "\n")

    # Record the full dehydrated model - Consider whether we want this.
    # results_handler.write("Model configuration:" + str(model.get_config()) + "\n")
    # Record a summary of the model
    model.summary(print_fn=lambda x: results_handler.write(x + '\n'))

    results_handler.write("\n")
    results_handler.close()


record_results()


def plot():
    """
    Plots the results.
    :return: No return value.
    """
    avg_acc_history = [np.mean([x[epoch] for x in acc_histories]) for epoch in range(num_epochs)]
    avg_val_acc_history = [np.mean([x[epoch] for x in val_acc_histories]) for epoch in range(num_epochs)]
    import matplotlib.pyplot as plt
    plt.plot(range(1, len(avg_acc_history) + 1), avg_acc_history, 'bo', label='training acc')
    plt.plot(range(1, len(avg_val_acc_history) + 1), avg_val_acc_history, 'b', label='Validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.show()
