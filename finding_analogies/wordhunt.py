from nltk.corpus import gutenberg
from boyer_moore import find_boyer_moore_all_paras
from boyer_moore import find_any_patterns
from analogy_strings import analogy_string_list
from utils import para_to_pretty
from utils import sent_to_pretty
from personal import root as root
from sentence_parser import *
import csv

SOURCE_NAME = "GUT"

txt_file_name = "analogy_sentences_GUT.txt"
csv_file_name = "analogy_names_GUT.csv"
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
