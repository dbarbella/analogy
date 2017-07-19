import nltk
from analogy_strings import analogy_string_list
from sentence_parser import get_speech_tags
from personal import root
#------------------------
from nltk.classify import SklearnClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import sys
#------------------------
import random
import re
from boyer_moore import find_boyer_moore

def get_list(filename):
    # Returns all training data as a list
    # File should be formatted as a text line followed '>' in the next line
    # before a new text line.
    list = []
    file = open(filename, "r", encoding = "utf-8")
    for line in file.readlines():
        if line[0] != '>' and line != "\n":
            final = line.split("]\",")[-1].split(".")[0].split("?")[0].split("!")[0]
            list.append(final)

    return list

# Returns all training data as a list which contains only the text(removing the source, paragraph #, sentence #, ratings
def get_list_re(filename):
    list = []
    file = open(filename, "r", encoding = "utf-8")
    for line in file.readlines():
        if line[0] != '>' and line != "\n":
            l = line.split("]\",")[-1]
            final = re.sub("[^a-zA-Z]"," ", l)
            list.append(final)

    return list

# preprocess the data so it can be used by the classifiers
def preprocess(samples, percent_test):
    num_samples = len(samples)
    random.shuffle(samples)
    cutoff = int((1.0 - percent_test) * num_samples)
    # create a train set and a test/development set
    feature_sets = [(text, label) for (text, label) in samples]
    train_set =  feature_sets[:cutoff]
    test_set = feature_sets[cutoff:]
    # separate the training data and the training labels
    train_data = [text for (text, label) in train_set]
    train_labels = [label for (text, label) in train_set]
    # separate the test data and the test labels
    test_data = [text for (text, label) in test_set]
    test_labels = [label for (text, label) in test_set]
    return(train_data, train_labels, test_data, test_labels)

# Transform the data so it can be represented by prepositional phrases.
def preposition(train_data, test_data):
    pp_training_set = [get_pp(text) for text in train_data]
    pp_test_set = [get_pp(text) for text in test_data]
    dict_vect = DictVectorizer()
    PpTrans = dict_vect.fit_transform(pp_training_set)
    PpTest = dict_vect.fit(pp_test_set)
    return (PpTrans, PpTest)

# Transform the data so it can be represented using tfidf
def tfidf(train_data, test_data):
    TfidfVect = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False)
    TfidfTrans = TfidfVect.fit_transform(train_data)
    TfidfTrans_test = TfidfVect.transform(test_data)
    return(TfidfTrans, TfidfTrans_test)

# Transform the data so it can be represented using Count Vectorizer
def countvect(train_data, test_data):
    CountVect = CountVectorizer(lowercase=False)
    CountTrans = CountVect.fit_transform(train_data)
    CountTest = CountVect.transform(test_data)
    return(CountTrans, CountTest)

# Transform the data so it can be represented using Hashing Vectorizer
def hashing(train_data, test_data, classifier):
    if classifier == "naive":
        HashVect = HashingVectorizer(lowercase=False, non_negative=True)
    else:
         HashVect = HashingVectorizer(lowercase=False)
    HashTrans = HashVect.fit_transform(train_data)
    HashTest = HashVect.transform(test_data)
    return(HashTrans, HashTest)

# Implementetion of the fmeasure metric, which calculates the precision, recall and f1measure given a confusion matrix
def fmeasure(matrix):
    value1 = (matrix[0][1] + matrix[0][0])
    value2 = (matrix[1][0] + matrix[0][0])
    if value1 == 0 or value2 == 0:
        precision = 0
        recall = 0
        f_measure = 0
    else:
        precision = matrix[0][0] / value1
        recall = matrix[0][0] / value2
        if precision == 0:
            f_measure = 0
        else:
            f_measure = (2 * precision * recall) / (precision + recall)
    return(precision, recall, f_measure)

def classify(train_data, train_labels, test_data, test_labels, classifier_name, representation, extra=[]):
    clfier = get_classifier(classifier_name, extra)
    train_set, test_set = get_data(train_data, test_data, representation, classifier)

    learn_results = clfier.fit(train_set, train_labels)
    score = learn_results.score(test_set, test_labels)
    test_predict = learn_results.predict(test_set)
    matrix = confusion_matrix(test_labels, test_predict, labels = ['YES', 'NO'])
    precision, recall, f_measure = fmeasure(matrix)

    return (score, matrix, precision, recall, f_measure)

def get_classifier(name, extra):
    if name == "svm":
        if extra == "" or extra == "svc":
            return SVC()
        elif extra == "linear":
            return LinearSVC()
        elif extra == "nusvc":
            return NuSVC()
    elif name == "neural":
        return MLPClassifier()
    elif name == "naive":
        return MultinomialNB()
    elif name == "max_ent":
        return LogisticRegression()
    else:
        sys.exit("This classifier has not been implemented yet.")
        return None

def get_data(train_data, test_data, representation, classifier):
    if representation == "tfidf":
        return tfidf(train_data, test_data)
    elif representation == "count":
        return countvect(train_data, test_data)
    elif representation == "hash":
        if classifier == "naive":
            return hashing(train_data, test_data, naive)
        else:
            return hashing(train_data, test_data, naive)
    else:
        sys.exit("This representation has not been implemented yet.")
        return None
