import random
import pandas as pd
from get_path import get_path
import csv
from sys import argv
from nltk import download, word_tokenize, pos_tag
import fnmatch
import os


#uncomment the following two lines if nltk gives an error
#download("punkt")
#download('averaged_perceptron_tagger')


in_path = argv[1]
out_path = argv[2]

NUM_PER_FILE = int(argv[3])
NUM_OF_FILES = int(argv[4])
NUM_OF_BOOKS = 59510

if out_path[-1] != "/":
    out_path += "/"

    
def like_is_verb(sentence):
  toks = word_tokenize(sentence)
  word_pos_pairs = pos_tag(toks)
  if sentence.lower().count('like') != 1:
    return False
  for word_tag in word_pos_pairs:
    word = word_tag[0]
    tag = word_tag[1]
    if word == 'like':
    #if it is some form of verb
    #the first two letters will be VB
      if tag[0:2] == 'VB':
        return True
  return False



def get_random_analogy():
    book_id = random.randint(0,NUM_OF_BOOKS)
    path = get_path(book_id,in_path)
    analogies = path + "book%s_analogies.csv" % book_id
    #if the randomly generated path does not exist
    try:
        data = pd.read_csv(analogies)
    except:
        return get_random_analogy()
    #if file selected has no analogies in it, call the function again to pick a different analogy
    if len(data)  == 0:
        return get_random_analogy()
    #pick a random analogy    
    analogy = data.sample(n=1, random_state=1)
    #check if analogy is nanalogy
    while like_is_verb(str(analogy['text'])):
        return get_random_analogy()
    return analogy

def get_random_analogies(num):
    analogies = []
    #faster than taking lenght of analogies every time
    len_analogies = 0
    while len_analogies < num:
        analogy = get_random_analogy()
        length = len(analogy)
        if length < 10 or length > 30:
            pass
        analogies.append(analogy)
        len_analogies += 1
    #returns a list of pandas dataframes
    return analogies


def write_random_analogies(num, out_name):
    list_analogies = get_random_analogies(num)
    frame_analogies = pd.concat(list_analogies, ignore_index=True, sort=False)
    frame_analogies.to_csv(out_name, encoding='utf-8', index=False)
    
def main():
    last_csv_file = len(fnmatch.filter(os.listdir(out_path), '*.csv'))
    begin = last_csv_file
    end = last_csv_file + NUM_OF_FILES
    for i in range(begin, end):
        out_name = out_path + "analogy_%s.csv" % i
        write_random_analogies(NUM_PER_FILE, out_name)
    

main()
#get_random_analogy()



