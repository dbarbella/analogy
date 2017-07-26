from main_interface import *
from emails import send_email
import traceback

list_of_classifiers = ["naive", "svm", "max_ent", "neural"]
list_of_representation = ["count", "tfidf", "hash"]
positive_set = 'test_extractions/bc_samples.txt'
negative_set = 'test_extractions/bc_grounds.txt'
extras = {}
trial_info = 'Positive Analogy File: ' + positive_set + '\nNegative Analogy File: '+ negative_set + '\nExtra: ' + str(extras) + '\n'

errors = []

for classifier in list_of_classifiers:
    for representation in list_of_representation:
        try:
            print("Classifier: " + classifier + "\tRepresentation: " + representation)
            analogy_trial(positive_set, negative_set, .5, representation, classifier, extra=extras,timer= 5)
            print()
        except:
            e = sys.exc_info()
            print(e)
            errors.append((classifier, representation, e, traceback.format_exc()))
            print()
            

if errors:
    error_msg = trial_info
    for error in errors:
        error_msg = error_msg + '\nError in classifier: ' + str(error[0]) + '\nRepresentation: ' + str(error[1]) + '\nError msg: ' + str(error[3]) + "\n"
    print("-----------------------------------------------------------------")
    print("Errors:\n")
    print(error_msg)
    send_email(error_msg)
    
    
