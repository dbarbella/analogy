import nltk
from nltk.corpus import PlaintextCorpusReader
# corpus_root = '~/pysetup/corpora'
# wordlists = PlaintextCorpusReader(corpus_root, '.*')
corpus_root = 'corpora'
susdpcr = PlaintextCorpusReader(corpus_root, ['susd/susdfull.txt'])
fileids = susdpcr.fileids()
susdfullwords = susdpcr.words('susd/susdfull.txt')

print(fileids)
print(susdfullwords)
#print(susdpcr.raw())
# susdtext = nltk.Text(susdfullwords)