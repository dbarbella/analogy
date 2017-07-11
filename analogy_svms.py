import nltk
from analogy_strings import analogy_string_list
from sentence_parser import get_speech_tags
from personal import root
#------------------------
from nltk.classify import SklearnClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from nltk.classify import maxent
#------------------------
import random
import re
from boyer_moore import find_boyer_moore

# Implementation of the classifier to detect analogy.
# From http://www.nltk.org/book/ch06.html

analogy_file_name = "test_extractions/demo_analogy_samples.txt"
non_analogy_file_name = "test_extractions/demo_analogy_ground.txt"

def get_analogy_string2(text):
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

    return "none"

def get_analogy_string(text):
    # Returns a tuple of the first analogy indicator along with its speech tag in the text.
    # e.g. (('as', 'RB'), ('fast', 'RB'), ('as', 'IN'))
    result = []
    tokens = text.split()
    #tagged_text = nltk.pos_tag(tokens)

    for pattern in analogy_string_list:
        match_index = find_boyer_moore(tokens, pattern)
        if match_index != -1:
            for i in range(len(pattern)):
                result.append(tokens[match_index + i])
            return result

    return result

def get_all_analogy_string(text):
    # Returns a tuple of all analogy indicators along with its speech tag in the text.
    # e.g. (('as', 'RB'), ('fast', 'RB'), ('as', 'IN'))
    result = []
    tokens = text.split()
    tagged_text = nltk.pos_tag(tokens)

    for pattern in analogy_string_list:
        match_index = find_boyer_moore(tokens, pattern)
        if match_index != -1:
            for i in range(len(pattern)):
                result.append(tagged_text[match_index + i])

    return result

def get_list(filename):
    # Returns all training data as a list
    # File should be formatted as a text line followed '>' in the next line
    # before a new text line.
    list = []
    file = open(filename, "r", encoding = "utf-8")
    for line in file.readlines():
        if line[0] != '>' and line != "\n":
            list.append(line)

    return list


analogy_list = get_list(analogy_file_name)
non_analogy_list = get_list(non_analogy_file_name)

# labeled data.
samples = [(text, 'YES') for text in analogy_list] + [(text, 'NO') for text in non_analogy_list]
random.shuffle(samples)

# divide data into training set and test set
feature_sets = [(get_analogy_string(text), label) for (text, label) in samples]
# train groups
train_set =  feature_sets[: 100]
train_data = [text for (text, label) in train_set]
train_labels = [label for (text, label) in train_set]
# test groups
test_set = feature_sets[100 :]
test_data = [text for (text, label) in test_set]
test_labels = [label for (text, label) in test_set]

#----------------------------------------- Naive Bayes Classifier -------------------------------------------------------------
# train classifier with training set
#classifier = nltk.NaiveBayesClassifier.train(train_set)
# test classifier
#print(nltk.classify.accuracy(classifier, test_set))
#print(classifier.show_most_informative_features(5))

#--------------------------------------------- SVM Classifiers --------------------------------------------------------------

# Preparing the data
TfidfVect = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False)
TfidfTrans = TfidfVect.fit_transform(train_data)
TfidfTrans_test = TfidfVect.transform(test_data)
# LinearSVC
LinearSvc = SklearnClassifier(LinearSVC().fit(TfidfTrans, train_labels).score(TfidfTrans_test, test_labels))
# SVC
Svc = SklearnClassifier(SVC().fit(TfidfTrans, train_labels).score(TfidfTrans_test, test_labels))
# NuSVC
NuSvc = SklearnClassifier(NuSVC().fit(TfidfTrans, train_labels).score(TfidfTrans_test, test_labels))
print("LinearSvc: ", LinearSvc)
print("Svc: ", Svc)
print("NuSvc: ", NuSvc)

#-------------------------------------- Maximum Entropy Classifier ----------------------------------------------------------
#max_ent = nltk.classify.MaxentClassifier.train(train_set, 'GIS', trace=0, max_iter=1000)
#print(nltk.classify.accuracy(max_ent, test_set))
#print(mec.show_most_informative_features(5))

#---------------------------------------------- Errors ------------------------------------------------------------------------

# show the ones the classifier guessed wrong.
#errors = []
#test_texts = samples[100 :]
#for (text, label) in test_texts:
#    guess = classifier.classify(get_analogy_string(text))
#    if guess != label:
#        errors.append((label, guess, get_analogy_string(text), text))

#print("Correct label, Guess, Text:")
#for item in errors:
#    print(item)

