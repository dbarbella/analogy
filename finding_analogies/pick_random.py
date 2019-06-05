import random
import pandas as pd
from get_path import get_path
import csv
from sys import argv
from nltk import download, word_tokenize, pos_tag
from fnmatch import filter
from os import listdir
from fixes import fixes
from context import get_context, next_sent, prev_sent
import re

#called like:
#python pick_random.py analogy_path book_path out_oath analogies_per_file total_num_of_files
#uncomment the following two lines if nltk gives an error
#download("punkt")
#download('averaged_perceptron_tagger')

NUM_OF_BOOKS = 59000
NUM_PER_FILE = int(argv[4])
NUM_OF_FILES = int(argv[5])

analogy_path = argv[1]
book_path = argv[2]
out_path = argv[3]

if analogy_path[-1] != "/":
       analogy_path += "/"
if book_path[-1] != "/":
    book_path += "/"
if out_path[-1] != "/":
       out_path += "/"

# analogy_path = "../../gutenberg_scrapped_analogies/"
# book_path = "../../gutenberg_scraps/"
# out_path ="../../turking_analogies/"

# NUM_PER_FILE = int(argv[1])
# NUM_OF_FILES = int(argv[2])

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

def fix(x):
    for i in fixes:
        x = re.sub(i[0], i[1], str(x))
    return x

#returns one pandas dataframe
def get_random_analogy():
    book_id = random.randint(0,NUM_OF_BOOKS)
    path = get_path(book_id, analogy_path)
    analogies = path + "book%s_analogies.csv" % book_id
    #if the randomly generated path does not exist
    try:
        df = pd.read_csv(analogies)
    except:  
        #print("{} does not exist".format(book_id))
        return get_random_analogy()
    #fixes the punctuation
    df['text'] = df['text'].apply(lambda x: fix(x))
    #if file selected has no analogies in it, call the function again to pick a different analogy
    if len(df)  == 0:
       #print("no analogies here")
        return get_random_analogy()
    #pick a random analogy    
    analogy = df.sample(n=1, random_state=1)
    #check if analogy is nanalogy
    while like_is_verb(str(analogy['text'])):
        #print("like is a verb")
        return get_random_analogy()
    #adds the context to the dataframe
    #to change values about this, including how many sentences before and after and input directory,
    #go to context.py.
    name = analogy['name'].to_string(index = False)
    try:
        prev, after = get_context(name, book_path)
    except Exception as e:
       # print("get context failed because:")
        #print(e)
        return get_random_analogy()
    analogy.insert(1,'prev_sent',prev)
    analogy.insert(3, 'next_sent', after)
    print("analogy returned")
    return analogy

    
def get_random_analogies(num):
    print("get random analogies worked")
    analogies = []
    #faster than taking lenght of analogies every time
    len_analogies = 0
    while len_analogies < num:
        analogy_df = get_random_analogy()
        analogy_str = analogy_df['text'].to_string(index = False)
        analogy_length = len(analogy_str)
        if analogy_length < 15 or analogy_length > 120:
            continue
        analogies.append(analogy_df)
        len_analogies += 1
    return analogies


def write_random_analogies(num, out_name):
    list_analogies = get_random_analogies(num)
    frame_analogies = pd.concat(list_analogies, ignore_index=True, sort=False)
    frame_analogies = remove_doubles(frame_analogies)
    frame_analogies.to_csv(out_name, encoding='utf-8', index=False)
    print("one file has been finished")
    print("\n\n\n")
    
    
#removes analogies with the same name
def remove_doubles(df):
    ids = []
    for i in df["name"]:
        if i not in ids:
            ids.append(i)
        else:
            while i in ids:
                df = df[df.name != i]
                analogy = get_random_analogy()
                df.append(analogy)
                i = analogy["name"].item()
            ids.append(i)
    return df

def main():
    #to make sure names are consecutive and not overlapping
    last_csv_file = len(filter(listdir(out_path), '*.csv')) + 1
    begin = last_csv_file
    end = last_csv_file + NUM_OF_FILES
    for i in range(begin, end):
        out_name = out_path + "analogy_%s.csv" % i
        write_random_analogies(NUM_PER_FILE, out_name)


if __name__ == "__main__":
    main()

