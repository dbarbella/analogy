from __future__ import print_function
import functions
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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    #('tfidf', TfidfTransformer()),
    ('svc', SVC()),
])

parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'vect__strip_accents' : ('ascii', 'unicode', None),
    'vect__analyzer' : ('word', 'char', 'char_wb'),    
    'vect__stop_words' : ('english', None),
    'vect__min_df': (0.1, 0.25),                        
    'svc__kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),
    'svc__tol' : (1e-3, 1e-2)
}

if __name__ == "__main__":
    positive_set = 'test_extractions/bc_samples.txt'
    negative_set = 'test_extractions/bc_grounds.txt'
    analogy_list = functions.get_list_re(positive_set)
    non_analogy_list = functions.get_list_re(negative_set)
    samples = [(text, 'YES') for text in analogy_list] + [(text, 'NO') for text in non_analogy_list]
    train_data, train_labels, test_data, test_labels = functions.preprocess(samples, 0.5)
    
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

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
