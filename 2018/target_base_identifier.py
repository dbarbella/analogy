import c_command as c_comm

ANALOGY_KEYWORD = "like"
ANALOGY_LABEL = "PP"
#NEED DOTS AFTER EVERY SENTENCE!
TEST = ["a flying dog is like a man, and a widow is like a window.", "dave is just like charlie.", "love me like you never loved me before.", "a car that is like a real one is like a man in the pool.", "bouncing a ball is like touching the stars."]

#Given a sentence, returns a tuple of target list and base list.
#Ex: ([solar system, sound ][atom, ball])
def extract_base_and_target(sentence):
	sentence = sentence.lower() #lower case
	extractor = c_comm.Extract() #create extract object
	base, target = extractor.search(sentence, ANALOGY_KEYWORD, ANALOGY_LABEL) #find base and target

	return base, target

if __name__ == '__main__':
	for next_sent in TEST:
		print(next_sent)
		base, target = extract_base_and_target(next_sent)
		print("BASE: ", base, "TARGET: ", target)
