
# read csv function
import csv
def readCSV(fileName,r):
    sent = []
    with open(fileName) as file:
        readcsv = csv.reader(file, delimiter=',')
        for row in readcsv:
            sentence = row[r]
            sent.append(sentence)
    return sent


# Using our corpus


neg = readCSV('./verified_non_analogies.csv',1)
pos = readCSV('./sentences.csv',0)
str_pos = "".join([p.lower() for p in pos])
labels = [1]*len(pos) + [0] * len(neg)
texts = pos + neg


# Import corpus from amazon


import keras
path = keras.utils.get_file('nietzsche.txt',
                            origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
str_pos = open(path).read().lower()

print('Corpus length:', len(str_pos))


# preprocess


import numpy as np
maxlen = 15 # we allow 15 characters per sequence
step = 3 # how many characters we skip after each sequence
sentences = [] # store samples
next_chars = []  
for i in range(0, len(str_pos) - maxlen, step):
    sentences.append(str_pos[i: i+maxlen])
    next_chars.append(str_pos[i+maxlen])
print("Number of sequences", len(sentences))

chars = sorted(list(set(str_pos))) # all the characters in the corpus
print("Unique characters, ", len(chars)) 
char_indices = dict((char,  chars.index(char)) for char in chars) # a hashmap with character and its index in the "vocab"
print("Vectorization...")
x = np.zeros((len(sentences), maxlen, len(chars)), dtype = np.bool)
y = np.zeros((len(sentences), len(chars)), dtype = np.bool)
for i, sentence in enumerate(sentences): # initialize the input and output
    for t, char in enumerate(sentence):
        x[i,t,char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build model


from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# sampling function


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# train and generate


import random
import sys
for epoch in range(1, 100):
    print('epoch', epoch)
    model.fit(x, y, batch_size=128, epochs=1)
    start_index = random.randint(0, len(str_pos) - maxlen - 1)
    generated_text = str_pos[start_index: start_index + maxlen] #given a sample text, predict a sequence.
    print('\n--- Generating with seed: "' + generated_text + '"')
    for temperature in [0.5,1.0, 1.2]:
        print('\n------ temperature:', temperature)
        sys.stdout.write(generated_text)
        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.
            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]
            generated_text += next_char
            generated_text = generated_text[1:]
            sys.stdout.write(next_char)
