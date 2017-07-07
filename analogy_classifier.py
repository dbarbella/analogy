import nltk
from analogy_strings import analogy_string_list
from sentence_parser import get_speech_tags
from personal import root
import random
import re

# Implementation of the classifier to detect analogy.
# http://www.nltk.org/book/ch06.html

analogy_file_name = "\\test_extractions\\demo_analogy_samples.txt"
non_analogy_file_name = "\\test_extractions\\demo_analogy_ground.txt"

def analogy_string_feature(text):
    # Returns the first analogy string in the text if found one, empty string
    # otherwise.
    for item in analogy_string_list:
        pattern = " ".join(item)
        if pattern != "as as":
            if pattern in text: return {"analogy_string" : pattern}
        else:
            ptrn = "as .*\ as"
            match = re.search(ptrn, text)
            if match != None: return {"analogy_string": pattern}

    return {"analogy_string" : ""}

def get_list(filename):
    # Returns all training data as a list
    # File should be formatted as a text line followed '>' in the next line
    # before a new text line.
    list = []
    file = open(root + filename, "r", encoding = "utf-8")
    for line in file.readlines():
        if line[0] != '>':
            list.append(line)

    return list


analogy_list = get_list(analogy_file_name)
non_analogy_list = get_list(non_analogy_file_name)

# labeled data.
samples = [(text, 'YES') for text in analogy_list] + [(text, 'NO') for text in non_analogy_list]
random.shuffle(samples)

# divide data into training set and test set
feature_sets = [(analogy_string_feature(text), label) for (text, label) in samples]
train_set =  feature_sets[: 100]
test_set = feature_sets[100 :]

# train classifier with training set
classifier = nltk.NaiveBayesClassifier.train(train_set)

# test classifier

# s1 = "He talks like a person who knows everything."
# s2 = "If the rain stops now, I will be able to go to work."
# print(classifier.classify(analogy_string_feature(s1)))
# print(classifier.classify(analogy_string_feature(s2)))
#
print(nltk.classify.accuracy(classifier, test_set))
print(classifier.show_most_informative_features(5))

# show the results the classifier guessed wrong.
errors = []
test_texts = samples[100 :]
for (text, label) in test_texts:
    guess = classifier.classify(analogy_string_feature(text))
    if guess != label:
        errors.append((label, guess, text))

print("Correct label, Guess, Text:")
for item in errors:
    print(item)
