'''
Given a directory, compiles all of the .txt files in that directory
into our standard format.
Use:
python compile_folder.py directory_name output_file_name
The output is a .csv of all of the sentenes.
'''

import glob
import sys
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize

#reload(sys)
#sys.setdefaultencoding('utf8')

dir_name = sys.argv[1]
output_name = sys.argv[2]
# path = "./" + dir_name + "/*.txt"


def compile_to_output():
    path = dir_name + "/*.txt"
    text_files = glob.glob(path)
    output_handler = open(output_name, "w", encoding="utf8")
    for next_file in text_files:
        print(next_file)
        output_handler.write(formatted_file(next_file))
    output_handler.close()


# Figure out what tokenization system we've been using. Is there one?
# What I want is a list of paragraphs, each of which is a list of sentences.
# Right now, I have a list of sentences
# Make sure we're handling null paras appropriately
# Last sentences of paragraphs currently contain extra newlines.
# This is now also going to be responsible for killing bad characters.
def formatted_file(file_name):
    output = ""
    source_name = just_file_name(file_name)
    print("Processing file:", source_name)
    with open(file_name, encoding="utf8") as file_handler:
        paras = file_handler.readlines()
        para_num = 0
        for para in paras:
            if not para.isspace():
                sents_in_para = sent_tokenize(para)
                sent_num = 0
                for sent in sents_in_para:
                    output += "$"
                    output += build_index(source_name, para_num, sent_num)
                    output += "$, $"
                    output += remove_bads(sent.rstrip())
                    output += "$,\n"
                    sent_num += 1
                para_num += 1
    return output

def remove_bads(in_string):
    return in_string.replace('\x00', '')

# Builds this:
# [BRWN, PARA#4051, SENT#12]
def build_index(source_name, para_num, sent_num):
    return "[" + source_name + ", PARA#" + str(para_num) + ", SENT#" + str(sent_num) + "]"

def para_tokenize(para):
    "Tokenizes a paragraph like NLTK does"
    return tokenize_sentences_in_para(sent_tokenize(para))
    # return tokenize_sents(sent_tokenize(para))


def tokenize_sentences_in_para(split_para):
    return [word_tokenize(sent) for sent in split_para]


def just_file_name(path):
    return path.split("/")[-1]


if __name__ == "__main__":
    compile_to_output()
