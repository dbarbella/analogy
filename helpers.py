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
from timeout import timeout
#------------------------
import random
import re
from boyer_moore import find_boyer_moore
#-------------------------
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_classif

def get_function(name):
    if name == "svc":
        return(SVC())
    elif name == "linearsvc":
        return(LinearSVC())
    elif name == "nusvc":
        return(NuSVC())
    elif name == "naive":
        return(MultinomialNB())
    elif name == "maxEnt":
        return(LogisticRegression())
    elif name == "neural":
        return(MLPClassifier())
    elif name == "hash":
        return(HashingVectorizer())
    elif name == "count":
        return(CountVectorizer())
    elif name == "tfidf":
        return(TfidfVectorizer())
    
def get_parameters(name):
    if name == "svc":
        return(parameters_svc)
    elif name == "linearsvc":
        return(parameters_linearsvc)
    elif name == "nusvc":
        return(parameters_nusvc)
    elif name == "naive":
        return(parameters_naive)
    elif name == "maxEnt":
        return(parameters_maxent)
    elif name == "neural":
        return(parameters_neural)
    elif name == "hash":
        return(parameters_hash)
    elif name == "count":
        return(parameters_count)
    elif name == "tfidf":
        return(parameters_tfidf)

def merge_two_dicts(x, y):
    z = x.copy()   
    z.update(y)    
    return z

parameters_count = {
    'count__max_df': (0.5, 0.75, 0.8),
    'count__max_features': (None, 5000, 10000, 50000),
    'count__ngram_range': ((1, 1),(1, 2)),  # unigrams or bigrams
    #'count__strip_accents' : ('ascii', 'unicode', None),
    'count__analyzer' : ('word', 'char', 'char_wb'),    
    #'count__stop_words' : ('english', None),
    'count__min_df': (0.1, 0.2, 0.3)                        
}
    
parameters_tfidf = {
    'tfidf__max_df': (0.5, 0.75, 0.8),
    'tfidf__max_features': (None, 5000, 10000, 50000),
    'tfidf__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'tfidf__strip_accents' : ('ascii', 'unicode', None),
    'tfidf__analyzer' : ('word', 'char', 'char_wb'),    
    #'tfidf__stop_words' : ('english', None),
    #'tfidf__min_df': (0.1, 0.2, 0.3, 0.4),   
    #'tfidf__norm' : ('l1', 'l2', None),
    'tfidf__use_idf': (True, False)
}

parameters_hash = {
    'hash__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'hash__strip_accents' : ('ascii', 'unicode', None),
    'hash__analyzer' : ('word', 'char', 'char_wb'),    
    #'hash__stop_words' : ('english', None),
    'hash__norm' : ('l1', 'l2', None)
}


parameters_svc = {
    'svc__kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),
    'svc__tol' : (1e-4, 1e-3, 1e-2),
    'svc__degree' : (2,3,4),
    #'svc__probability' : (False, True),
    #'svc__shrinking' : (False, True),
    'svc__decision_function_shape' : ('ovo', 'ovr')
}

parameters_linearsvc = {
    'linearsvc__penalty' : ('l1', 'l2'),
    #'linearsvc__loss' : ('hinge', 'squared_hinge'),
    #'linearsvc__dual' : (True, False),
    'linearsvc__multi_class' : ('ovr', 'crammer_singer'),
    'linearsvc__tol' : (1e-5, 1e-4, 1e-3),
    'linearsvc__intercept_scaling' : (0.9,1,1.3),
    #'linearsvc__fit_intercept' : (False, True)
}

parameters_nusvc = {
    'nusvc__kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),
    'nusvc__nu' : (0.5,0.6,0.7),
    'nusvc__tol' : (1e-4, 1e-3),
    'nusvc__degree' : (2,3,4),
    #'nusvc__probability' : (False, True),
    #'nusvc__shrinking' : (False, True),
    'nusvc__decision_function_shape' : ('ovo', 'ovr')
}

parameters_naive = {
    #'naive__fit_prior' : (False, True)
}


parameters_maxent = {
    'maxEnt__penalty' : ('l1', 'l2'),
    #'maxEnt__dual' : (True, False),
    'maxEnt__tol' : (1e-5, 1e-4, 1e-3),
    #'maxEnt__fit_intercept' : (False, True),
    'maxEnt__intercept_scaling' : (0.8,0.9,1,1.2,1.3),
    'maxEnt__solver' : ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'),
    #'maxEnt__multi_class' : ('ovr', 'multinomial'),
    #'maxEnt__warm_start' : (False, True)
}

parameters_neural = {
    'neural__activation' : ('identity', 'logistic', 'tanh', 'relu'),
    'neural__solver' : ('lbfgs', 'sgd', 'adam'),
    'neural__learning_rate' : ('constant', 'invscaling', 'adaptive'),
    #'neural__shuffle' : (False, True),
    'neural__tol' : (1e-5, 1e-4, 1e-3),
    #'neural__warm_start' : (False, True),
    #'neural__early_stopping' : (False, True),
    'neural__epsilon' : (1e-9, 1e-8, 1e-7, 1e-6)
}

def generate_parameters(representation, classifier):
    part1 = get_parameters(representation)
    part2 = get_parameters(classifier)
    parameters = merge_two_dicts(part1, part2)
    return(parameters)
