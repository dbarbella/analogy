from nltk.corpus import brown
from boyer_moore import find_boyer_moore_all_paras
from boyer_moore import find_any_patterns
from analogy_strings import analogy_string_list
from utils import para_to_pretty
from utils import sent_to_pretty
from personal import root as root
from sentence_parser import *

SOURCE_NAME = "BRWN"

file_name = "analogy_sentences_test.txt"
output_handler = open(root + "test_extractions/" + file_name, "w", encoding="utf-8")

# csv_file = "names.csv"
# csv_handler = open()

brown_paras = brown.paras()
para_indices = find_any_patterns(brown_paras, analogy_string_list)

for para_index in para_indices:
    sentence_pos = get_analogy_sentence(brown_paras[para_index], analogy_string_list)
    sentence = sentence_pos[0]
    sent_index = sentence_pos[1]
    if sentence != '':
        id_tag = "[" + SOURCE_NAME + ", PARA#" + str(para_index) + ", SENT#" + str(sent_index) + "]"
        output_handler.write(id_tag)
        output_handler.write(sentence + "\n")

output_handler.close()
