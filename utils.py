# Contains utility functions for analogy processing

# Converts a list of sentences that are lists of words into an easy-to-read paragraph
def para_to_pretty(para):
	return " ".join((flatten(para)))

def flatten(in_list):
	return [item for sublist in in_list for item in sublist]