## Removes things in the analogy corpus from the full SUSD and writes a new output.

import re

def make_susd_output(output_file_name, root):   
    analogies_handler = open(root + "analogy-susd-samples.txt", "r", encoding="utf-8")
    fulltext_handler = open(root + "susd-full-paras.txt", "r", encoding="utf-8")
    output_handler = open(root + "testoutputs/" + output_file_name, "w", encoding="utf-8")
    analogies = analogies_handler.read().split(";----")
    out_text = fulltext_handler.read()
    for analogy in analogies:
        #print(analogy, file=output_handler)
        out_text = out_text.replace(analogy.strip(), "\n;----\n")
    out_text = preprocess_susd(out_text)
    print(out_text, file=output_handler)
    #print(out_text)

def preprocess_susd(text):
    chunk_counter = 1
    newtext = ""
    for line in text.split("\n"):
        if not re.match("h\:*", line):
            if line == "************************************************************************":
                newtext += (";----\n>susd-bg-" + str(chunk_counter) + "\n")
                chunk_counter += 1
            elif line == ";----":
                newtext += (";----\n>susd-bg-" + str(chunk_counter) + "\n")
                chunk_counter += 1
            else:
                newtext += (line + "\n")
    return newtext
    

if __name__ == "__main__":
    make_susd_output("test-output-01.txt", "/eccs/users/barbeda/pysetup/corpora/")
