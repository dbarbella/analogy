from nltk.corpus import brown
from boyer_moore import find_boyer_moore_all_paras
from boyer_moore import find_any_patterns
from analogy_strings import analogy_string_list
from utils import para_to_pretty
from personal import root as root

brown_paras = brown.paras()

#para_indices = find_boyer_moore_all_paras(brown_paras, ['is', 'like', 'a'])
para_indices = find_any_patterns(brown_paras, analogy_string_list)
print(para_indices)

output_file_name = "test_001.txt"
output_handler = open(root + "test_extractions/" + output_file_name, "w", encoding="utf-8")


for para_index in para_indices:
    print("--------", file = output_handler)
    print(para_to_pretty(brown_paras[para_index]), file = output_handler)

output_handler.close()