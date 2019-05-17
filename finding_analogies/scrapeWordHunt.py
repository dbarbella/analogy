import sys
sys.path.append("..")

from nltk.corpus import gutenberg, PlaintextCorpusReader
from boyer_moore import find_boyer_moore_all_paras
from boyer_moore import find_any_patterns
from analogy_strings import analogy_string_list
from utils import para_to_pretty
from utils import sent_to_pretty
from sentence_parser import *
from personal import root as root
import nltk
import csv


"""
this is wordHunt.py adapted for the output files of gutenberg_scrapper.py
It will give a csv of analogies, with what file and where in the file
the analogies are in.
The output will be stored in analogy/finding_analogies/extractions
"""

SOURCE_NAME = "GUT"
NUMBER_OF_FILES = 50000

#converts text file to a list of lists where the outer list are paragraphs
#and the inner list is sentences, split by words
def text_to_paras(book_id):
    #book_root = '../book%s.txt' % book_id
    #put in directory of the .txt files
    book_root = '../../../bookScraping'
    corpus = PlaintextCorpusReader(book_root, '.*')
    paragraphs = corpus.paras('book%s.txt' % book_id)
    return paragraphs

def write_analogies(book_id):

    book_id = str(book_id)
    txt_file_name = "analogy_sentences_book%s.txt" % book_id
    csv_file_name = "analogy_names_book%s.csv" % book_id
    output_handler = open(root + "extractions\\" + txt_file_name, "w", encoding="utf-8")

    # Find the indices of all paragraphs that contain the patterns as listed in
    # analogy_string_list
    paras = text_to_paras(book_id)
    para_indices = find_any_patterns(paras, analogy_string_list)
    ids = {}            # save sentences' ids in hash table to prevent duplicates.

    # Extract the exact sentences and write them to csv and txt files.
    with open("extractions/large_gutenberg\\" + csv_file_name, 'w', encoding="utf-8") as csvfile:
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

# for i in range(11,101):
#     write_analogies(i)

write_analogies(10)
