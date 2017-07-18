from analogy_strings import analogy_string_list
from nltk.parse import stanford
import nltk
from nltk.stem.wordnet import WordNetLemmatizer

parser = StanfordParser()

def is_noun(label):
    noun_labels = {"NN", "NNP", "NNPS", "NNS", "NP"}
    return label in noun_labels

def is_verb(label):
    verb_labels = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
    return label in verb_labels

def is_adj(label):
    adj_labels = {"JJ", "JJR", "JJS"}
    return label in adj_labels

def convert_to_base_form(word, type):
    # Convert a verb or adjective to its baseform.
    # type should be either 'v' for verb or 'a' for adjective.
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
    result = []

    for tree in parser.parse(sentence.split()):
        subtrees = tree.subtrees()
        for subtree in subtrees:
            if subtree.label() == tag:
                result.append(subtree)

    return result

def get_pp(text):
    # Return: a list of prepositions inside PP's in
    # the text. If the phrase is preceded by a VP/ADJP, the result
    # include the verb/adj also. If the phrase is preceded by a NP,
    # the noun is not included.
    result = []

    for structure in parser.parse(nltk.word_tokenize(text)):
        tree = ParentedTree.convert(structure)
        for subtree in tree.subtrees():
            if subtree.label() == "PP":
                preposition = subtree.leaves()[0]
                left_sibling = subtree.left_sibling()
                left_sibling_label = left_sibling.label()
                if is_noun(left_sibling_label):
                    result.append(preposition)
                elif is_verb(left_sibling_label):
                    verb = convert_to_base_form(" ".join(left_sibling.leaves()), 'v')
                    word = verb + " " + preposition
                    result.append(word)
                elif is_adj(left_sibling_label):
                    adj = convert_to_base_form(" ".join(left_sibling.leaves()), 'a')
                    word = adj + " " + preposition
                    result.append(word)

    return {"prepositional_phrases" : result}
