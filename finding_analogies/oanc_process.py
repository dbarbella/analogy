from nltk.corpus.reader.tagged import TaggedCorpusReader
import glob
from nn.keras
/boyer_moore import find_boyer_moore_all_paras
from boyer_moore import find_any_patterns
from analogy_strings import analogy_string_list
from utils import para_to_pretty
from utils import sent_to_pretty
from personal import root as root
from sentence_parser import *
import csv

root = "S:/Users/David/Documents/Code/analogy"

def build_file_list(root_directory):
    '''
    Use glob to make a list of all of the .txt files in the directory of interest, recursively.
    :param root_directory:
    :return:
    '''
    file_list = glob.glob(root_directory + "/**/*.txt", recursive=True)
    return file_list


# Use the ANC tool to build an nltk version of the data here.
oanc_directory = root + "\\corpora\\oanc\\nltk-data"  # oanc/nltk-data"
oanc_files = build_file_list(oanc_directory)

corpus = TaggedCorpusReader('.', ["./ch1.txt"])
print(corpus.fileids())
x = corpus.words()[:50]
print(x)
print("Step one complete.")

# See http://www.nltk.org/howto/corpus.html
corpus2 = TaggedCorpusReader(oanc_directory, oanc_files, sep="_")  # We need to specify that _ is used as a separator.
print(corpus2.fileids())
x = corpus2.words()[:50]
print(x)
y = corpus2.paras()[:10]
for i in range (len(y)):
    print(len(y[i]))





"""
This script is an alternative/demo to scrapeWordHunt.py, but is not
used in this folder.
"""
SOURCE_NAME = "OANC"

txt_file_name = "analogy_sentences_OANC.txt"
csv_file_name = "analogy_names_OANC.csv"
output_handler = open(root + "extractions\\" + txt_file_name, "w", encoding="utf-8")

# Find the indices of all paragraphs that contain the patterns as listed in
# analogy_string_list
paras = gutenberg.paras()
para_indices = find_any_patterns(paras, analogy_string_list)
ids = {}            # save sentences' ids in hash table to prevent duplicates.
# Extract the exact sentences and write them to csv and txt files.
with open(root + "extractions\\" + csv_file_name, 'w', encoding="utf-8") as csvfile:
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
            id_tag = "[" + SOURCE_NAME + ", PARA#" + str(para_index) + ", SENT#" + str(sent_index) + "]"
            if not id_tag in ids.keys():
                ids[id_tag] = True
                output_handler.write(id_tag + "\n")
                output_handler.write(sentence + "\n")
                writer.writerow({'name': id_tag, 'text': sentence})

output_handler.close()

#  print(corpus.words()[:30])
