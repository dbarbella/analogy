from analogy_strings import analogy_string_list
import nltk

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
