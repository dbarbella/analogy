<<<<<<< HEAD
# This produces a corpus made up of all of the sentences in the brown corpus that are not
# included in verified_analogies.csv or verified_non_analogies.csv

import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
from personal import root

import nltk
from nltk.corpus import brown

import re
# What we want to create is a file of all of the brown sentences. As we go, we should
# not include the ones that are in our labeled set.
# The code we have that names sentences from that corpus is at wordhunt.py 


# Extract the exact sentences and write them to csv and txt files.
def make_unlabeled_set(labeled_set_files, storage_file):
    samples_to_exclude = []
    # From the list of labeled set files, get the items to exclude
    for next_file in labeled_set_files:
        with open(root + "corpora/" + next_file) as next_file_handler:
            for line in next_file_handler:
                ptrn = '"\\[.*\\]"'
                match = re.findall(ptrn, line)
                if match:
                    samples_to_exclude.append(match[0])
    # Go through the brown corpus, and if we're not looking at something in the list
    # of things to exclude, output it to at .csv
    # This is very inefficient, but is most foolproof and we shouldn't have to run this
    # many times.
    brown_paras = brown.paras()
    para_index, sent_index = 0, 1
    SOURCE_NAME = "BRWN"
    with open(root + "test_extractions/" + storage_file, 'w') as storage_file_handler:
        for para in brown_paras:
            for sent in para:
                id_tag = '"[' + SOURCE_NAME + ", PARA#" + str(para_index) + ", SENT#" + str(sent_index) + ']"'
                if id_tag not in samples_to_exclude:
                    print(id_tag + "," + " ".join(sent),file = storage_file_handler)
                sent_index += 1
            para_index += 1
            sent_index = 1
    
=======
# This produces a corpus made up of all of the sentences in the brown corpus that are not
# included in verified_analogies.csv or verified_non_analogies.csv

import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
from personal import root

import nltk
from nltk.corpus import brown

import re
# What we want to create is a file of all of the brown sentences. As we go, we should
# not include the ones that are in our labeled set.
# The code we have that names sentences from that corpus is at wordhunt.py 


# Extract the exact sentences and write them to csv and txt files.
def make_unlabeled_set(labeled_set_files, storage_file):
    samples_to_exclude = []
    # From the list of labeled set files, get the items to exclude
    for next_file in labeled_set_files:
        with open(root + "corpora/" + next_file) as next_file_handler:
            for line in next_file_handler:
                ptrn = '"\\[.*\\]"'
                match = re.findall(ptrn, line)
                if match:
                    samples_to_exclude.append(match[0])
    # Go through the brown corpus, and if we're not looking at something in the list
    # of things to exclude, output it to at .csv
    # This is very inefficient, but is most foolproof and we shouldn't have to run this
    # many times.
    brown_paras = brown.paras()
    para_index, sent_index = 0, 1
    SOURCE_NAME = "BRWN"
    with open(root + "test_extractions/" + storage_file, 'w') as storage_file_handler:
        for para in brown_paras:
            for sent in para:
                id_tag = '"[' + SOURCE_NAME + ", PARA#" + str(para_index) + ", SENT#" + str(sent_index) + ']"'
                if id_tag not in samples_to_exclude:
                    print(id_tag + "," + " ".join(sent),file = storage_file_handler)
                sent_index += 1
            para_index += 1
            sent_index = 1
    
>>>>>>> 4657b10db19ea4bc1a12a2cac842e4988eb39836
make_unlabeled_set(["verified_analogies.csv", "verified_non_analogies.csv"],"dmb_open_test.csv")