from main_interface import *
from emails import send_email

list_of_classifiers = ["naive", "svm", "max_ent", "neural"]
list_of_representation = ["count", "tfidf", "hash"]
positive_set = 'test_extractions/bc_samples.txt'
negative_set = 'test_extractions/bc_grounds.txt'
errors = []
for classifier in list_of_classifiers:
    for representation in list_of_representation:
        try:
            print("Classifier: " + classifier + "\tRepresentation: " + representation)
            analogy_trial(positive_set, negative_set, .5, representation, classifier, timer= 5)
            print()
        except:
            e = sys.exc_info()
            print(e)
            errors.append((classifier, representation, e))
            print()

if errors:
    error_msg = ''
    for error in errors:
        error_msg = error_msg + '\nError in classifier: ' + str(error[0]) + '\nRepresentation: ' + str(error[1]) + '\nError msg: ' + str(error[2]) + "\n"
    print("-----------------------------------------------------------------")
    print("Errors:\n")
    print(error_msg)
    send_email(error_msg)
    
    