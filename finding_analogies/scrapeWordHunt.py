import sys
sys.path.append("..")

from nltk.corpus import gutenberg, PlaintextCorpusReader
from boyer_moore import find_boyer_moore_all_paras
from boyer_moore import find_any_patterns
from analogy_strings import analogy_string_list
from utils import para_to_pretty
from utils import sent_to_pretty
from get_path import get_path 
from sentence_parser import get_analogy_sentence
from subprocess import call, Popen
from time import sleep
from sys import argv
import nltk
import csv


"""
this is wordHunt.py adapted for the output files and directory hierarchy of gutenberg_scrapper.py
It will give both .txt and a csv of analogies, with what file and where in the file
the analogies are in.
The output will be stored in the indicated output directory. The program willc reate subdirectories for the texts.
It's ran as:
python scrapeWordHunt.py input_dir output_dir begin_num end_num
"""

SOURCE_NAME = "GUT"
NUMBER_OF_FILES = 50000

#the first command line argument is the directory where the books are located
book_dir = argv[1]
#the second command line argument is the output directory
out_dir = argv[2]

begin = int(argv[3])
end = int(argv[4])

if out_dir[-1] != "/":
    out_dir +=  "/"
if book_dir[-1] != "/":
    book_dir +=  "/"
    
#converts text file to a list of lists where the outer list are paragraphs
#and the inner list is sentences, split by words
def text_to_paras(book_id):
    #put in directory of the .txt files
    book_path = get_path(book_id, book_dir)
    corpus = PlaintextCorpusReader(book_path, '.*')
    paragraphs = corpus.paras('book%s.txt' % book_id)
    return paragraphs

def write_analogies(book_id):
    
    out_path = get_path(book_id, out_dir)
    book_id = str(book_id)
    txt_file_name = out_path 
    csv_file_name = out_path
    output_handler = open(txt_file_name + "book%s_analogies.txt" % book_id, "w", encoding="utf-8")
    # Find the indices of all paragraphs that contain the patterns as listed in analogy_string_list
    paras = text_to_paras(book_id)
    para_indices = find_any_patterns(paras, analogy_string_list)
    ids = {}      # save sentences' ids in hash table to prevent duplicates.

    # Extract the exact sentences and write them to csv and txt files.
    with open(csv_file_name + "book%s_analogies.csv" % book_id, 'w', encoding="utf-8") as csvfile:
        fieldnames = ['name', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, lineterminator='\n')
        writer.writeheader()
        for para_index in para_indices:
            sentence_pos = get_analogy_sentence(paras[para_index], analogy_string_list)
            # get_analogy_sentence returns a 2-element tuple. The first element is the analogy string,
            # the second is its sentence index within the paragraph.
            sentence = sentence_pos[0]
            sent_index = sentence_pos[1]
            if sentence != '':
                # Generate the ID of the sentence (e.g. [BRWN, PARA#1, SENT#1]).
                id_tag = "[" + book_id + ", PARA#" + str(para_index) + ", SENT#" + str(sent_index) + "]"
                if not id_tag in ids.keys():
                    ids[id_tag] = True
                    output_handler.write(id_tag + "\n")
                    output_handler.write(sentence + "\n")
                    writer.writerow({'name': id_tag, 'text': sentence})
    output_handler.close()


if __name__ == "__main__":
    #uncomment these if it is the first time running the code with a certain output directory
    #Popen(["bash", "-c", "chmod +x make_dirs.sh"])
    #call(["bash","./make_dirs.sh",out_dir])
    #sleep(1)  
    for i in range(begin, end):
        #the book might not exist, which will raise an error.
        try:
            write_analogies(i)
            print(("book%s" % i) + " worked")
        except Exception as e:
            print(e)
            pass 

