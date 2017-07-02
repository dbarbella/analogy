import nltk
# from nltk.book import *
# nltk.download()
from nltk.corpus import brown
from boyer_moore import find_boyer_moore_all
from boyer_moore import find_boyer_moore_all_paras

# A list of the sentences, each of which is a list of the words in the sentence
# brown_sents = brown.sents(categories=['news', 'editorial', 'reviews'])
# A list of the words
#brown_words = brown.words(categories=['news', 'editorial', 'reviews'])
brown_words = brown.words()
brown_paras = brown.paras()
brown_sents = brown.sents()
# The text, but marked up with POS information
# brown_raw = brown.raw(categories=['news', 'editorial', 'reviews'])
#print(brown_sents[0:10])
#print(brown_words[0:100])
#print(brown_raw[0:100])

brown_text = nltk.Text(brown_words)
'''
print(brown_words[:1000])
print("----------")
print(brown_text[:1000])

print(brown_paras[:3])
print("---------")
print(brown_sents[:3])
'''
para_indices = find_boyer_moore_all_paras(brown_paras, ['is', 'like', 'a'])
print(para_indices)
for para_index in para_indices:
    print(brown_paras[para_index])
#bm_indicies = find_boyer_moore_all(brown_words, ['is', 'like', 'a'])
#print(bm_indicies)
#print(len(brown_words))
'''
for fileid in gutenberg.fileids():
     num_chars = len(gutenberg.raw(fileid))
     num_words = len(gutenberg.words(fileid))
     num_sents = len(gutenberg.sents(fileid))
     num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
     print(round(num_chars/num_words), round(num_words/num_sents), round(num_words/num_vocab), fileid)
     '''
'''
from nltk.corpus import treebank
# This is only two sentences
# How do we get this to actually do parsing?
##t = treebank.parsed_sents('wsj_0001.mrg')[0]
##for i in range(10):
##    print(i)
##    print(treebank.parsed_sents('wsj_0001.mrg')[i])

sentence = "A cat is like a bicycle."
tokens = nltk.word_tokenize(sentence)
tagged = nltk.pos_tag(tokens)
print(tagged)

fcp2 = nltk.parse.load_parser('grammars/book_grammars/feat0.fcfg')
'''
'''
sentence = "A cat is like a bicycle."
tokens = nltk.word_tokenize(sentence)
tagged = nltk.pos_tag(tokens)
print(tagged)

fcp2 = nltk.parse.load_parser('grammars/book_grammars/feat0.fcfg')
'''