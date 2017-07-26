from analogy_strings import analogy_string_list
from nltk.parse import stanford
import nltk
from nltk.tree import ParentedTree
from nltk.stem.wordnet import WordNetLemmatizer
from personal import root
import functions

noun_labels = {"NN", "NNP", "NNPS", "NNS", "NP"}
adj_labels = {"JJ", "JJR", "JJS"}

def is_noun(label):
    return label in noun_labels

def is_verb(label):
    verb_labels = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
    return label in verb_labels

def is_adj(label):
    return label in adj_labels

def prepositions_set(filename):
    prepositions = set()
    prepositions_file = open(filename, "r")
    for line in prepositions_file:
        prepositions.add(line[:-1])
    return prepositions

# prepositions = prepositions_set("prepositions.txt")

def is_conventional_preposition(word):
    return word.lower() in prepositions

def convert_to_base_form(word, type):
    # Convert a verb or adjective to its baseform.
    # type should be either 'v' for 'verb' or 'a' for 'adjective'.
    wnl = WordNetLemmatizer()
    return wnl.lemmatize(word, type)

def get_analogy_sentence(para, pattern_list):
    # Gets the sentence that contains the analogy phrases as specified
    # in the analogy_string_list from a paragraph and its index within the para.
    # The paragraph must be in NLTK format.
    for i in range(len(para)):
        for item in pattern_list:
            pattern = " ".join(item)
            whole_sentence = " ".join(para[i])
            if whole_sentence.find(pattern) != -1:
                return [whole_sentence, i + 1]
    return ['', -1]

def get_speech_tags(sentence):
    # Returns speech tags of a string.
    # Based on http://www.nltk.org/book/ch05.html
    result = []
    text = nltk.word_tokenize(sentence)
    tagged_sent = nltk.pos_tag(text)

    for item in tagged_sent:
        result.append(item[1])
    return result[:-1]

def get_subtree(sentence, tag):
    # Returns a list of subtrees of the sentence with the specified tag.
    parser = stanford.StanfordParser()
    result = []

    for tree in parser.parse(sentence.split()):
        subtrees = tree.subtrees()
        for subtree in subtrees:
            if subtree.label() == tag:
                result.append(subtree)

    return result

def get_pp_old(text):
    # Return: a list of prepositions inside PP's in
    # the text. If the phrase is preceded by a VP/ADJP, the result
    # include the verb/adj also. If the phrase is preceded by a NP,
    # the noun is not included.
    phrases = {}

    for structure in parser.parse(nltk.word_tokenize(text)):
        tree = ParentedTree.convert(structure)
        for subtree in tree.subtrees():
            if subtree.label() == "PP":
                preposition = subtree.leaves()[0]
                left_sibling = subtree.left_sibling()

                if left_sibling != None:
                    left_sibling_label = left_sibling.label()
                    if is_noun(left_sibling_label):
                        phrases[preposition] = True
                    elif is_verb(left_sibling_label):
                        verb = convert_to_base_form(" ".join(left_sibling.leaves()), 'v')
                        word = verb + " " + preposition
                        phrases[word] = True
                    elif is_adj(left_sibling_label):
                        adj = convert_to_base_form(" ".join(left_sibling.leaves()), 'a')
                        word = adj + " " + preposition
                        phrases[word] = True

    return phrases

def get_pp(text):
    # Return: a list of prepositions inside PP's in
    # the text. If the phrase is preceded by a VP/ADJP, the result
    # include the verb/adj also. If the phrase is preceded by a NP,
    # the noun is not included.
    parser = stanford.StanfordParser()
    phrases = {}

    for structure in parser.parse(nltk.word_tokenize(text)):
        tree = ParentedTree.convert(structure)
        for subtree in tree.subtrees():
            if subtree.label() == "PP":
                preposition = subtree.leaves()[0]       # first word of the prep phrase
                left_sibling = subtree.left_sibling()
                if left_sibling != None:
                    left_sibling_label = left_sibling.label()
                    if is_adj(left_sibling_label):
                        adj = " ".join(left_sibling.leaves())
                        preposition = adj + " " + preposition
                if not is_conventional_preposition(preposition):
                    phrases[preposition] = True

    return phrases
