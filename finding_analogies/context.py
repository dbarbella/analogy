from nltk.corpus import gutenberg, PlaintextCorpusReader
from nltk.tokenize import sent_tokenize
import sys
from get_path import get_path
from fixes import fixes
import re
from collections import deque
sys.path.append("..")

#book_dir = "../../gutenberg_scrap_backup/gutenberg_scrap/"
#book_dir = "../../testing_context/"
#TEST_NAME = "[450, PARA#1560, SENT#1]"
#Number of sentences before the analogy and after the analogy to be selected
NUM_PREV = 1
NUM_NEXT = 1

#Fixes punctuation. This is important as it makes sure NLTK manages to tokenize sentences.
def fix(x):
    for i in fixes:
        x = re.sub(i[0], i[1], str(x))
    return x

#returns paragraph index and sentence index
def get_id(name):
    para = "[0-9]+"
    id_nums = re.findall(para, name)
    para = int(id_nums[1])
    sent = int(id_nums[2])
    return para, sent

#finds numbers in the id, which are book_no, para_no and sentence_no 
def get_book_id(name):
    nums = "[0-9]+"
    book_id = re.findall(nums, name)[0]   
    return book_id

#takes a name and returns
def text_to_paras(name, book_dir):
    book_id = get_book_id(name)
    book_path = get_path(book_id, book_dir)
    corpus = PlaintextCorpusReader(book_path, '.*')
    paragraphs = corpus.paras('book%s.txt' % book_id)
    return paragraphs

def join_paragraph(paras, index):
    sent = ""
    para = paras[index]
    for i in range(len(para)):
        sent += " ".join(para[i])
    sent = fix(sent)
    sentences = sent_tokenize(sent)
    return sentences

#gives the previous sentences after an analogy
#if there are not enough sentences, it gives what is available.
#takes name of an analogy and nltk corpusized text as arguments
def prev_sent(paras, name):
    prev_sents = ""
    para_index, sent_index = get_id(name)
    para = join_paragraph(paras, para_index)
    analogy = para[sent_index - 1]
    #sometimes paragraphs are short, in which case there won't be enough sentences in the paragrap
    
    while sent_index - 1 - NUM_PREV < 0:
        para_index -= 1
        try:
            para = join_paragraph(paras, para_index) + para
            sent_index = para.index(analogy) 
        except Exception as e:
            print(e)
            break
    sent_index = para.index(analogy) 
    prev_sents = para[sent_index - NUM_PREV:sent_index]
    #if you don't join pandas will thrown an error when the length of list is less than NUM_PREV
    return " ".join(prev_sents)
                          
#gives the next sentences after an analogy
#if there are not enough sentences, it gives what is available.
def next_sent(paras, name):
    next_sents = ""
    para_index, sent_index = get_id(name)
    para = join_paragraph(paras, para_index)
    #we don't need the sentences before the analogy sentence here
    analogy = para[sent_index - 1]
    #para = para[sent_index - 2::]
    #sometimes paragraphs are short, in which case there won't be enough sentences in the paragraph
    while len(para) < sent_index + NUM_NEXT:
        para_index += 1
        #in case there aren't enough sentences left in the text.
        try:
            para = para + join_paragraph(paras, para_index) 
            sent_index = para.index(analogy)
        except:
#             next_sents = para[0:sent_index + 2 + NUM_NEXT]
#             return next_sents
            break
    sent_index = para.index(analogy)
    next_sents = para[sent_index + 1:sent_index + 1 + NUM_NEXT]
    #if you don't join pandas will thrown an error when the length of list is less than NUM_NEXT
    return " ".join(next_sents)

def get_context(name, book_dir):
    paras = text_to_paras(name, book_dir)
    prev = prev_sent(paras, name)
    after = next_sent(paras, name)
    return prev, after

TEST_NAME = "[10, PARA#1564, SENT#1]"
book_dir = "../../gutenberg_scraps/"
analogy_path = "../../gutenberg_scrapped_analogies/"


def main():
    prev, after = get_context(TEST_NAME, book_dir)
    print("prev", prev)
    print("\n\n")
    print("after",after)
#main()
