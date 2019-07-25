#!/usr/bin/env python
# coding: utf-8

#phi - simple LSTM

# Use:
# python Keras_RNN.py non_analogies_csv analogies_csv glove_file
# Or just
# python Keras_RNN.py to run with things in the default locations.

import sys
import csv
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, LSTM

def readCSV(fileName,r):
    sent = []
    with open(fileName) as file:
        readcsv = csv.reader(file, delimiter=',')
        for row in readcsv:
            # This seems like a workaround for our source files being
            # inconsistent - revisit this.
            sentence = row[r]
            sent.append(sentence)
    return sent

default_non_analogies = '../.././corpora/verified_non_analogies.csv'
default_sentences = '../.././corpora/sentences.csv'
default_glove_directory = '../.././corpora/glove/'
default_glove_file = 'glove.6B.100d.txt'

num_args = len(sys.argv)


if num_args < 2:
    non_analogies_file = default_non_analogies 
else:
    non_analogies_file = sys.argv[1]

if num_args < 3:
    sentences_file = default_sentences 
else:
    sentences_file = sys.argv[2]
    
if num_args < 4:
    glove_file = default_glove_directory + default_glove_file
else:
    glove_file = sys.argv[3]


neg = readCSV(non_analogies_file,1)
pos = readCSV(sentences_file,0)
labels = [1]*len(pos) + [0] * len(neg)
texts = pos + neg





maxlen = 28 #maximum allowed number of words in a sentence 

max_words = 10000#choosing the most 10000 common words
test = 40 #number of testing samples

tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print("Found {} unique tokens".format(len(word_index)))

data = pad_sequences(sequences, maxlen = maxlen) # pad sequences to a fixed length
labels = np.asarray(labels)

indices = np.arange(data.shape[0])

np.random.shuffle(indices) #shuffle the sentences

data = data[indices]
labels = labels[indices]
# train test split
train_data = data[:len(data)-test]
test_data = data[len(data)-test:]
train_targets = labels[:len(data) - test]
test_targets = labels[len(data)-test:]

print("Shape of data ", data.shape)
print("Shape of label", labels.shape)
print("Shape of train_data", train_data.shape)
print("Shape of test_data", test_data.shape)
print("Shape of train_targets", train_targets.shape)
print("Shape of test_targets", test_targets.shape)

embedding_index = {}

#download glove before this
f = open(glove_file)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype = 'float32')
    embedding_index[word] = coefs
f.close()

print("Found {} words".format(len(embedding_index)))


k = 4 # 4 k-fold
num_val_samples = len(train_data) // k
num_epochs = 10


# Embed sentences

embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector




def build_model():
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length = maxlen))
    model.add(LSTM(32))
    model.add(Dense(1, activation = 'sigmoid'))
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False
    model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
    model.save_weights('pre_trained_glove_model.h5')
    return model

#k-fold training
val_acc_history = []
acc_history = []
for i in range(k):
    print('processing fold: #',i)
    val_data = train_data[i * num_val_samples: (i+1)*num_val_samples]
    val_targets = train_targets[i * num_val_samples : (i+1)*num_val_samples]
    
    partial_train_data = np.concatenate(
    [train_data[:i*num_val_samples],
    train_data[(i+1)*num_val_samples:]], axis = 0)
    
    partial_test_data = np.concatenate(
    [train_targets[:i*num_val_samples],
    train_targets[(i+1)*num_val_samples:]], axis = 0)
    
    model = build_model()
    history = model.fit(partial_train_data, partial_test_data, epochs = num_epochs, batch_size = 1, verbose = 0,
                       validation_data = (val_data, val_targets))
    val_acc = history.history['val_acc']
    acc = history.history['acc']
    val_acc_history.append(val_acc)
    acc_history.append(acc)


#plot results in training

def plot():
    avg_acc_history = [np.mean([x[i] for x in acc_history] ) for i in range(num_epochs)]
    avg_val_acc_history = [np.mean([x[i] for x in val_acc_history] ) for i in range(num_epochs)]
    import matplotlib.pyplot as plt
    plt.plot(range(1, len(avg_acc_history) + 1), avg_acc_history, 'bo', label = 'training acc')
    plt.plot(range(1, len(avg_val_acc_history) + 1), avg_val_acc_history, 'b', label = 'Validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.show()



results = model.evaluate(test_data, test_targets)


