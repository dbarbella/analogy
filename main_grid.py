from __future__ import print_function
import functions
import helpers
import random
import sys
from export_log import outputResults
#--------
import timeout 
import math
from pprint import pprint
from time import time
import logging
#--------
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

def accuracy_from_matrix(matrix):
    total = matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1]
    correct = matrix[0][0] + matrix[1][1]
    accuracy = correct / total
    return(accuracy)

def main():
    positive_set = 'test_extractions/bc_samples.txt' #'test_extractions/test-neural-hash-samples.txt' 
    negative_set = 'test_extractions/bc_grounds.txt' #'test_extractions/test-neural-hash-ground.txt' 
    analogy_list = functions.get_list_re(positive_set)
    non_analogy_list = functions.get_list_re(negative_set)
    samples = [(text, 'YES') for text in analogy_list] + [(text, 'NO') for text in non_analogy_list]
    train_data, train_labels, test_data, test_labels = functions.preprocess(samples, 0.5)
    pipeline = []
    classifiers = ['svc', 'linearsvc', 'nusvc', 'naive', 'maxEnt', 'neural']
    classifiers2 = ['neural']
    representations = ['tfidf', 'count', 'hash']
    representations2 = ['hash']
    
    for classifier in classifiers:
        for representation in representations:
            pipeline = (Pipeline([(representation, helpers.get_function(representation)),
                          (classifier, helpers.get_function(classifier)),]))
            parameters = helpers.generate_parameters(representation, classifier)
            grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, error_score=-1)
            print("Performing grid search...")
            print("pipeline:", [name for name, _ in pipeline.steps])
            print("parameters:")
            pprint(parameters)
            t0 = time()
            grid_search.fit(train_data, train_labels)
            print("done in %0.3fs" % (time() - t0))
            print()

            print("Best score: %0.3f" % grid_search.best_score_)
            print("Best parameters set:")
            best_parameters = grid_search.best_estimator_.get_params()
            for param_name in sorted(parameters.keys()):
                print("\t%s: %r" % (param_name, best_parameters[param_name]))
            print()
            
            print("Getting the confusion matrix for the best estimator:")
            prediction = grid_search.best_estimator_.predict(test_data)
            matrix = confusion_matrix(test_labels, prediction, labels = ['YES', 'NO'])
            precision, recall, f_measure = functions.fmeasure(matrix)
            accuracy = accuracy_from_matrix(matrix)
            print("Accuracy ", accuracy)
            print("Precision, recall, f-score:")
            print(precision, recall, f_measure)
            print(matrix)
            print()

main()
