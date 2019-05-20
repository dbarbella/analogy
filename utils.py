from personal import root as root
# Contains utility functions for analogy processing

# Converts a list of sentences that are lists of words into an easy-to-read paragraph
def para_to_pretty(para):
	return " ".join((flatten(para)))

def flatten(in_list):
	return [item for sublist in in_list for item in sublist]

# Takes in one of the analogy corpus flatfiles and returns a list of tuples of the titles and their text
def analogy_ff_to_list(file_loc):
    file_handler = open(file_loc, "r")
    split_file = file_handler.read().split(";----")
    file_handler.close()
    samples = [_snip_corpus_name(sample) for sample in split_file  if _snip_corpus_name(sample)]
    return samples

def sent_to_pretty(sentence):
	return " ".join(sentence)

# Takes something of the form "/n>name/ntexttexttext..." and returns (name, texttexttext...)
def _snip_corpus_name(in_string):
    stripped_left = in_string.lstrip()
    split_string = stripped_left.split("\n",1)
    if len(split_string) > 1:
        return (split_string[0],split_string[1].lstrip())
    else:
        print ("Something went wrong:\n" + str(split_string))


#print(analogy_ff_to_list(root + "/corpora/analogy-samples.txt"))
