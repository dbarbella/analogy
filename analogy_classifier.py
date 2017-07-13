import nltk
from analogy_strings import analogy_string_list
from sentence_parser import get_speech_tags
from personal import root
import random
import re
from boyer_moore import find_boyer_moore

# Implementation of the classifier to detect analogy.
# From http://www.nltk.org/book/ch06.html

analogy_file_name = "\\test_extractions\\analogy_sentences.txt"
non_analogy_file_name = "\\test_extractions\\demo_analogy_ground.txt"

def get_analogy_string2(text):
    # Returns the first analogy string in the text if found one, empty string
    # otherwise.s
    for item in analogy_string_list:
        pattern = " ".join(item)
        if pattern != "as as":
            if pattern in text: return {"analogy_string" : pattern}
        else:
            ptrn = "as .*\ as"
            match = re.search(ptrn, text)
            if match != None: return {"analogy_string": pattern}

    return {"analogy_string" : ""}

def get_analogy_string(text):
    # Returns a tuple of the first analogy indicator along with its speech tag in the text.
    # e.g. (('as', 'RB'), ('fast', 'RB'), ('as', 'IN'))
    result = []
    tokens = text.split()
    tagged_text = nltk.pos_tag(tokens)

    for pattern in analogy_string_list:
        match_index = find_boyer_moore(tokens, pattern)
        if match_index != -1:
            for i in range(len(pattern)):
                result.append(tagged_text[match_index + i])
            return {"analogy_indicator:" : tuple(result)}

    return {"analogy_indicator:" : tuple(result)}

def get_all_analogy_string(text):
    # Returns a tuple of all analogy indicators along with its speech tag in the text.
    # e.g. (('as', 'RB'), ('fast', 'RB'), ('as', 'IN'))
    result = []
    tokens = text.split()
    tagged_text = nltk.pos_tag(tokens)

    for pattern in analogy_string_list:
        match_index = find_boyer_moore(tokens, pattern)
        if match_index != -1:
            for i in range(len(pattern)):
                result.append(tagged_text[match_index + i])

    return {"analogy_indicator:" : tuple(result)}

def get_pos_tags(text):
    tokens = text.split()
    return {"pos_tags:" : tuple(nltk.pos_tag(tokens))}

def get_list(filename):
    # Returns all training data as a list
    # File should be formatted as a text line followed '>' in the next line
    # before a new text line.
    list = []
    file = open(root + filename, "r", encoding = "utf-8")
    for line in file.readlines():
        if line[0] != '[':
            list.append(line[:-1])

    return list


analogy_list = get_list(analogy_file_name)
non_analogy_list = get_list(non_analogy_file_name)

# labeled data.
samples = [(text, 'YES') for text in analogy_list] + [(text, 'NO') for text in non_analogy_list]
random.shuffle(samples)

# divide data into training set and test set
feature_sets = [({"text:" : text}, label) for (text, label) in samples]
mid_point = int(len(samples) / 2)
train_set =  feature_sets[: mid_point]
test_set = feature_sets[mid_point :]

# train classifier with training set
classifier = nltk.NaiveBayesClassifier.train(train_set)

# test classifier

# Use the classifier to classify s1 and s2:
# s1 = "He talks like a person who knows everything."
# s2 = "If the rain stops now, I will be able to go to work."
# print(classifier.classify(get_analogy_string(s1)))
# print(classifier.classify(get_analogy_string(s2)))

print(nltk.classify.accuracy(classifier, test_set))
print(classifier.show_most_informative_features(5))

# show the ones the classifier guessed wrong.
# errors = []
# test_texts = samples[mid_point :]
# for (text, label) in test_texts:
#     guess = classifier.classify(get_pos_tags(text))
#     if guess != label:
#         errors.append((label, guess, get_pos_tags(text), text))
#
# print("Correct label, Guess, Text:")
# for item in errors:
#     print(item)
