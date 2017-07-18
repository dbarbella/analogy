import functions
import random
import sys

# positive_set is the set of positive examples, as a file
# negative_set is the set of negative examples, as a file
# percent_test is the portion of the imput sets that should be used as the test set
# The rest will be the training set.
# representation is the representation to use, as a string
# classifier is the classifier to use, as a string
# extra is other information that is used to specify the behavior of the classifier
def analogy_trial(positive_set, negative_set, percent_test, representation, classifier, extra=[]):
    # Read in the set of positive examples
    analogy_list = functions.get_list_re(positive_set)
    # Read in the set of negative examples
    non_analogy_list = functions.get_list_re(negative_set)
    # Randomly divide them into a training set and a test set
    samples = [(text, 'YES') for text in analogy_list] + [(text, 'NO') for text in non_analogy_list]
    # Run classifier, generate results based on the value passed in for representation
    if classifier == "svm":
        train_data, train_labels, test_data, test_labels = functions.preprocess(samples, percent_test)
        return(functions.svm(train_data, train_labels, test_data, test_labels, representation, extra))
    elif classifier == "neural":
        train_data, train_labels, test_data, test_labels = functions.preprocess(samples, percent_test)        
        return(functions.neural(train_data, train_labels, test_data, test_labels, representation))
    elif classifier == "naive":
        train_data, train_labels, test_data, test_labels = functions.preprocess(samples, percent_test)
        return(functions.naive(train_data, train_labels, test_data, test_labels, representation))
    elif classifier == "max_ent":
        train_data, train_labels, test_data, test_labels = functions.preprocess(samples, percent_test)
        return(functions.max_ent(train_data, train_labels, test_data, test_labels, representation))
    else:
        sys.exit("This classifier has not been implemented yet.")
    # Store results
