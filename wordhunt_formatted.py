'''
This is similar to wordhunt.py, but for text files formatted
in our standard format
'''

import sys
from analogy_strings import analogy_string_list
from boyer_moore import find_boyer_moore_lists
import csv

input_file = sys.argv[1]
pos_output_file = sys.argv[2]
neg_output_file = sys.argv[3]

pos_output_handler = open(pos_output_file, "w", encoding="utf-8")
neg_output_handler = open(neg_output_file, "w", encoding="utf-8")

# Each line of the input file should look like this:
#
# "[sourcename.txt, PARA#1, SENT#4]", "That just burns my toast, you know what I mean?",

def produce_output():
    with open(input_file, "r", encoding="utf-8") as input_handler:
        # This is not currently handling quoted text correctly.
        # That's not surprising, I think? It's probably interpreting the quotes as close quotes
        # We should pick some other quote character.
        csv_reader = csv.reader(input_handler, skipinitialspace=True, quotechar='$')
        csv_writer_pos = csv.writer(pos_output_handler, quotechar='$', quoting=csv.QUOTE_ALL)
        csv_writer_neg = csv.writer(neg_output_handler, quotechar='$', quoting=csv.QUOTE_ALL)
        # This needs to be able to deal with bad characters, like NULL
        # Or maybe don't let those be included in the first place?
        for row in csv_reader:
            #out_form = ", ".join(row) + "\n"
            # This needs to deal with quotes - maybe?
            # This should use csv.writer
            print(row[0])
            if basic_wordhunt_filter(row[1].strip(), analogy_string_list):
                csv_writer_pos.writerow(row)
            else:
                #print("Neg")
                csv_writer_neg.writerow(row)


def basic_wordhunt_filter(sentence, pattern_list):
    '''
    Uses a basic set of words to check to see if a sentence is a potential analogy.
    This is not a comprehensive detection method; it is a first pass at finding
    data that might be useful.
    '''
    # Rewrite this so it doesn't return in the middle
    # This needs to handle ['as', '', 'as'] - We may want Boyer Moore after all?
    # Look for more sophisticated pattern matching, perhaps.
    low_sentence = sentence.lower()
    #print(low_sentence)
    for pattern in pattern_list:
        # in doesn't work here. Let's try something else.
        if find_boyer_moore_lists(low_sentence.split(), pattern) >= 0:
            return True
    return False



if __name__ == "__main__":
    print("Starting...")
    produce_output()
    print("Done.")

pos_output_handler.close()
neg_output_handler.close()
