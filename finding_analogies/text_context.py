import nltk
from get_path import get_path
from context import next_sent, prev_sent, get_context
from random import randint 
import sys
import pandas as pd
from fixes import fixes
import re
sys.path.append("..")
#this is not working yet

def fix(x):
    for i in fixes:
        x = re.sub(i[0], i[1], str(x))
    return x

def test(book_id):
    analogy_path =  get_path(book_id, "../../gutenberg_scrapped_analogies/") + "book{}_analogies.csv".format(book_id)
    book_path = get_path(book_id, "../../gutenberg_scraps/") + "book{}.txt".format(book_id)
    analogies = pd.read_csv(analogy_path)
    analogies['text'] = analogies['text'].apply(lambda x: fix(x))
    with open(book_path, 'r') as book:
        text = book.read().replace("\n",' ')
        text = fix(text)
        sentences = nltk.sent_tokenize(text)
    count = 0
    for i in analogies['text']:
        print(i)
#         prev, after = get_context(i['name'])
#         print(prev)
#         print(i['text'])
#         print(after)
        
        
#test(16)