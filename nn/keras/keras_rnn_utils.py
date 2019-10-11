#!/usr/bin/env python
# coding: utf-8

import csv
import numpy as np
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense
from sklearn.metrics import confusion_matrix
import time


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
    with open(file_name, encoding="utf8") as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            print("row:", row)
            sentence = row[sentence_column]
            sent.append(sentence)
    return sent


def produce_glove_embedding_index(glove_file):
    """
    :param glove_file: A glove file
    :return: A dictionary that maps words to np arrays of values.
    """
    index = {}
    f = open(glove_file, encoding="utf8")
    for line in f:
        values = line.split()
        next_word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        index[next_word] = coefs
    f.close()
    return index


def produce_glove_embedding_matrix(lexicon_size, embedding_dim, embedding_index, word_index):
    """
    :param lexicon_size: Use the n most common words
    :param embedding_dim: The number of embedding dimensions to use. It doesn't make sense for this to exceed the
    number of dimensions of the glove file, probably.
    :param embedding_index: A dictionary that maps words to numpy arrays of numbers
    :param word_index: A dictionary that maps words in the source sentences to unique indices.
    :return: embedding_matrix, a matrix of embedding vectors for the lexicon_size most frequent words in the input
    sentences, in order.
    """
    embedding_matrix = np.zeros((lexicon_size, embedding_dim))
    # Iterate through each word, index pair.
    for word, i in word_index.items():
        # Check to see if it is one of the lexicon_size most frequent.
        if i < lexicon_size:
            # If it is, get the embedding vector for that word.
            embedding_vector = embedding_index.get(word)
            # If we found one, make that the correct column of the embedding matrix
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                print("Warning: No embedding vector found for: ", word)
    return embedding_matrix


def first(x):
    return x[0]


def equals_one(x):
    return x == 1


def evaluate_model(model, test_data, test_labels):
    # Actually does the evaluation, using the test data.
    # The results returned by this are a list of two things:
    # A loss and an accuracy.
    results = model.evaluate(test_data, test_labels)
    '''
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(model.metrics_names)
    print(results)
    '''
    test_loss = results[0]
    test_accuracy = results[1]
    # How can we get the confusion matrix?
    # Predicting the Test set results? Should this be model.predict? What should we predict on?
    # y_pred = classifier.predict(X_test)
    y_pred = model.predict(test_data)
    # check to make sure we're rounding this in the right direction.
    # Actually change of heart. Make these all into booleans.
    # y_pred = map(first, (y_pred > .5))
    y_pred_bools = (y_pred > .5)
    y_pred = list(map(first, y_pred_bools))
    bool_test_labels = list(map(equals_one, test_labels))
    results_confusion_matrix = confusion_matrix(bool_test_labels, y_pred)
    return test_loss, test_accuracy, results_confusion_matrix


def train_model(num_val_samples, train_data, train_labels, lexicon_size, embedding_dim, max_sen_length_in_words,
                embedding_matrix, num_epochs, epoch_batch_size, num_folds):
    val_acc_histories = []
    acc_histories = []
    for i in range(num_folds):

        # print('processing fold: #', i)
        # Get the training data from the i'th fold
        # Validation is the i'th slice every time. Training is everything else
        val_data = train_data[(i * num_val_samples):((i + 1) * num_val_samples)]
        val_targets = train_labels[(i * num_val_samples):((i + 1) * num_val_samples)]
        # This is everything that's not in the validation set.
        partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                             train_data[(i + 1) * num_val_samples:]], axis=0)
        partial_train_labels = np.concatenate([train_labels[:i * num_val_samples],
                                               train_labels[(i + 1) * num_val_samples:]], axis=0)

        # Put together the structure of the ANN
        model = build_model(lexicon_size, embedding_dim, max_sen_length_in_words, embedding_matrix)

        # Trains the model. Learn about fit here: https://keras.io/models/sequential/#fit
        # This returns a history object, which we probably want to store somewhere, or pull things out of.
        history = model.fit(partial_train_data, partial_train_labels, epochs=num_epochs, batch_size=epoch_batch_size,
                            verbose=0, validation_data=(val_data, val_targets))
        # Validation accuracy and training accuracy
        val_acc = history.history['val_acc']
        acc = history.history['acc']
        # Append these to lists.
        val_acc_histories.append(val_acc)
        acc_histories.append(acc)

    return model, val_acc_histories, acc_histories


def build_model(lexicon_size, embedding_dim, max_sen_length_in_words, embedding_matrix):
    # This is keras.models's Sequential()
    model = Sequential()
    # Adds an embedding layer - See https://keras.io/layers/embeddings/
    # The first argument to the embedding layer is the size of the vocabulary. The is the dimensions
    # of the input.
    # Second argument is the dimensions of the embedding space. This is the dimensions of the output.
    # input_length is the maximum length of the sentence, in words.
    model.add(Embedding(lexicon_size, embedding_dim, input_length=max_sen_length_in_words))
    # Adds an LSTM layer of size 32?
    model.add(LSTM(32))
    # Adds a dense layer
    model.add(Dense(1, activation='sigmoid'))
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    model.save_weights('pre_trained_glove_model.h5')
    # model.summary()
    return model


