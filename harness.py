from main_interface import *
from emails import send_email

list_of_classifiers = ["naive", "svm", "max_ent", "neural"]
list_of_representation = ["count", "tfidf", "hash"]
positive_set = 'test_extractions/bc_samples.txt'
negative_set = 'test_extractions/bc_grounds.txt'
errors = []
try:
    for classifier in list_of_classifiers:
        for representation in list_of_representation:
            print("Classifier: " + classifier + "\tRepresentation: " + representation)
            analogy_trial(positive_set, negative_set, .5, representation, classifier, timer= 5)
            print()
except:
    e = sys.exc_info()
    error.append(e)

if errors:
    error_msg = '\n'.join(erros)
    send_email(error_msg)
    