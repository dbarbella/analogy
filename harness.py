from main_interface import *

list_of_classifiers = ["naive", "svm", "max_ent", "neural"]
list_of_representation = ["count", "tfidf", "hash"]
positive_set = 'test_extractions/bc_samples.txt'
negative_set = 'test_extractions/bc_grounds.txt'
    
for classifier in list_of_classifiers:
    for representation in list_of_representation:
        print("Classifier: " + classifier + "\tRepresentation: " + representation)
        analogy_trial(positive_set, negative_set, .5, representation, classifier, {"sub_class":""}, 5)
        print()