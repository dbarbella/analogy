# import nltk
# from analogy_strings import analogy_string_list
# from sentence_parser import get_speech_tags
# from personal import root
#------------------------
from nltk.classify import SklearnClassifier
from sklearn.feature_extraction import DictVectorizer
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
from timeout import timeout
#------------------------
import random
import re
import csv
from boyer_moore import find_boyer_moore
#-------------------------
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_classif
sys.path.insert(0, './2018')
from DependencyParsing import dependency_parse

"""To add new feature, add new function that takes a train and a test set and extra and return a train and a test set based on what representation you want ot use"""
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
    sent = []
    with open(filename) as file:
        readcsv = csv.reader(file, delimiter=',')
        for row in readcsv:
            sentence = row[1]
            sent.append(sentence)
    return sent

# preprocess the data so it can be used by the classifiers
def preprocess(samples, percent_test, caller=''):
    num_samples = len(samples)
    if caller == 'test_main_interface_output': 
        random.seed(1234)
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
    PpTest = dict_vect.transform(pp_test_set)
    return (PpTrans, PpTest)

def base_target_pair(train_data, test_data, extra):
    bt_training = [dependency_parse(text) for text in train_data]
    bt_testing = [dependency_parse(text) for text in test_data]
    dict_vect = DictVectorizer()
    bt_train = dict_vect.fit_transform(bt_training)
    bt_test = dict_vect.transform(bt_testing)
    return (bt_train, bt_test)


# Transform the data so it can be represented using tfidf
def tfidf(train_data, test_data, extra):
    TfidfVect = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False, stop_words=extra['stop_words'], max_df=extra['max_df'], norm=extra['norm'], min_df=extra['min_df'])
    TfidfTrans = TfidfVect.fit_transform(train_data)
    TfidfTrans_test = TfidfVect.transform(test_data)
    return(TfidfTrans, TfidfTrans_test)

# Transform the data so it can be represented using Count Vectorizer
def countvect(train_data, test_data, extra):
    CountVect = CountVectorizer(lowercase=False, stop_words=extra['stop_words'], max_df=extra['max_df'])
    CountTrans = CountVect.fit_transform(train_data)
    CountTest = CountVect.transform(test_data)
    return(CountTrans, CountTest)

# Transform the data so it can be represented using Hashing Vectorizer
def hashing(train_data, test_data,extra, classifier=[]):
    if classifier == "naive":
        HashVect = HashingVectorizer(lowercase=False, non_negative=True, stop_words=extra['stop_words'],norm=extra['norm'])
    else:
         HashVect = HashingVectorizer(lowercase=False, stop_words=extra['stop_words'], norm=extra['norm'])
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
        precision = matrix[0][0] / value2
        recall = matrix[0][0] / value1
        if precision == 0:
            f_measure = 0
        else:
            f_measure = (2 * precision * recall) / (precision + recall)
    return(precision, recall, f_measure)

def classify(train_data, train_labels, test_data, test_labels, classifier_name, representation, extra={"sub_class":""}, time=1000000000):
    @timeout(time)
    def _classify(train_data, train_labels, test_data, test_labels, classifier_name, representation, extra):
        clfier = get_classifier(classifier_name, extra)
        train_set, test_set = get_representation(train_data, test_data, representation, classifier_name, extra)
        learn_results = clfier.fit(train_set, train_labels)
        score = learn_results.score(test_set, test_labels)
        test_predict = learn_results.predict(test_set)
        matrix = confusion_matrix(test_labels, test_predict, labels = ['YES', 'NO'])
        precision, recall, f_measure = fmeasure(matrix)
        return (score, matrix, precision, recall, f_measure)
    return _classify(train_data, train_labels, test_data, test_labels, classifier_name, representation, extra)


def get_classifier(name, extra):
    if name == "svm":
        if extra["sub_class"] == "" or extra["sub_class"] == "svc":
            return SVC(kernel=extra['kernel'], max_iter=extra['max_iter_svc'], )
        elif extra["sub_class"] == "linear":
            return LinearSVC(C=extra['C'])
        elif extra["sub_class"] == "nusvc":
            return NuSVC(kernel=extra['kernel'], max_iter=extra['max_iter_svc'], decision_function_shape=extra['decision_function_shape'], degree=extra['degree'], nu=extra['nu'], tol=extra['tol'])
    elif name == "neural":
        return MLPClassifier(hidden_layer_sizes=extra['hidden_layer_sizes'], activation=extra['activation'], solver=extra['solver'], max_iter=extra['max_iter'], early_stopping=extra['early_stopping'], learning_rate=extra['learning_rate'])
    elif name == "naive":
        return MultinomialNB(alpha=extra['alpha'])
    elif name == "max_ent":
        return LogisticRegression(C=extra['C'], max_iter=extra['max_iter_log'], solver=extra['solver_log'])
    else:
        sys.exit("This classifier has not been implemented yet.")
        return None

def get_representation(train_data, test_data, representation, classifier, extra):
    if representation == "tfidf":
        return tfidf(train_data, test_data, extra)
    elif representation == "count":
        return countvect(train_data, test_data, extra)
    elif representation == "base_target":
        return base_target_pair(train_data,test_data,extra)
    elif representation == "hash":
        if classifier == "naive":
            return hashing(train_data, test_data, extra, "naive")
        else:
            return hashing(train_data, test_data, extra)
    else:
        sys.exit("This representation has not been implemented yet.")
        return None

def set_default(extra, key, value):
    try:
        if extra[key] == "":
            extra[key] = value
    except KeyError:
        extra[key] = value

def set_extra(extra):
    set_default(extra,'sub_class', "")
    set_default(extra,'stop_words', None)
    set_default(extra,'hidden_layer_sizes', 100)
    set_default(extra,'activation', 'relu')
    set_default(extra,'max_df', 1.0) #1.0
    set_default(extra,'min_df', 0.5) #1.0
    set_default(extra,'norm', 'l2')
    set_default(extra,'alpha', 1.0)
    set_default(extra,'kernel', 'rbf')
    set_default(extra,'max_iter', 200)
    set_default(extra,'max_iter_log', 100)
    set_default(extra,'max_iter_linear', 1000)
    set_default(extra,'max_iter_svc', -1)
    set_default(extra,'solver', 'adam')
    set_default(extra,'solver_log', 'liblinear')
    set_default(extra,'early_stopping', False)
    set_default(extra,'C', 1.0)
    set_default(extra,'degree', 3)
    set_default(extra,'learning_rate','constant')
    set_default(extra,'decision_function_shape','ovr')
    set_default(extra,'nu', 0.5)
    set_default(extra,'tol', 0.001)
    set_default(extra, 'word_hypernyms', 0.001)
    return(extra)


def classify_pipeline(train_data, train_labels, test_data, test_labels, classifier_name, representation, extra={"sub_class":""}, time=1000000000):
    @timeout(time)
    def _classify(train_data, train_labels, test_data, test_labels, classifier_name, representation, extra):
        clfier = get_classifier(classifier_name, extra)
        train_set, test_set = get_representation(train_data, test_data, representation, classifier_name, extra)
        
        selector = SelectPercentile(f_classif, percentile=10)
        estimators = [('reduce_dim', selector), ('clf', clfier)]
        pipe = Pipeline(estimators)
        
        learn_results = pipe.fit(train_set, train_labels)
        score = learn_results.score(test_set, test_labels)
        test_predict = learn_results.predict(test_set)
        matrix = confusion_matrix(test_labels, test_predict, labels = ['YES', 'NO'])
        precision, recall, f_measure = fmeasure(matrix)
        return (score, matrix, precision, recall, f_measure)
    return _classify(train_data, train_labels, test_data, test_labels, classifier_name, representation, extra)
