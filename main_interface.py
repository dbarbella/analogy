import analogy_svms
import random

# positive_set is the set of positive examples, as a file
# negative_set is the set of negative examples, as a file
# percent_test is the portion of the imput sets that should be used as the test set
# The rest will be the training set.
# representation is the representation to use, as a string
# classifier is the classifier to use, as a string
def analogy_trial(positive_set, negative_set, percent_test, representation, classifier):
    # Read in the set of positive examples
    analogy_list = analogy_svms.get_list(analogy_file_name)
    # Read in the set of negative examples
    non_analogy_list = analogy_svms.get_list(non_analogy_file_name)
    # Randomly divide them into a training set and a test set
    samples = [(text, 'YES') for text in analogy_list] + [(text, 'NO') for text in non_analogy_list]
    num_samples = len[samples]
    random.shuffle(samples)
    cutoff = int((1.0 - percent_test) * num_samples)
    training_samples = samples[:cutoff]
    test_samples = samples[cutoff:]
    # Generate a representation, based on the value passed in for representation

    # Run