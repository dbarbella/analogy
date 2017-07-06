from analogy_strings import analogy_string_list
import nltk

def get_analogy_sentence(para, pattern_list):
    # Gets the sentence that contains the analogy phrases as specified
    # in the analogy_string_list from a paragraph. Returns "" if no such phrases
    # found.
    # The paragraph must be in NLTK format.
    for sentence in para:
        for item in pattern_list:
            pattern = " ".join(item)
            whole_sentence = " ".join(sentence)
            if whole_sentence.find(pattern) != -1:
                return sentence
    return ""

def get_speech_tags(sentence):
    # Returns speech tags of a string.
    # Based on http://www.nltk.org/book/ch05.html
    result = []
    text = nltk.word_tokenize(sentence)
    tagged_sent = nltk.pos_tag(text)

    for item in tagged_sent:
        result.append(item[1])
    return result[:-1]
