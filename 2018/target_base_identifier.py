import sys
import c_command as c_comm
import utilities

ANALOGY_KEYWORD = "like"
ANALOGY_LABEL = "PP"
#NEED DOTS AFTER EVERY SENTENCE!
TEST = ["a flying dog is like a man, and a widow is like a window.", "dave is just like charlie.", "love me like you never loved me before.", "a car that is like a real one is like a man in the pool.", "bouncing a ball is like touching the stars.","a flying dog is the same as a man" ]

#Given a sentence, returns a tuple of target list, base list and action list.
#Ex: ([solar system, sound ][atom, ball][behaves, bounces])
def extract_base_and_target(sentence, signals):
	sentence = sentence.lower() #lower case

	#replace signals with ANALOGY_KEYWORD='like'
	if "like" not in sentence:
		for signal in signals:
			if signal in sentence:
				sentence = sentence.replace(signal, ANALOGY_KEYWORD)

	extractor = c_comm.Extract() #create extract object
	base, target, action = extractor.search(sentence, ANALOGY_KEYWORD, ANALOGY_LABEL) #find base, target and action

	return base, target, action

#converts a list into a string.
def produce_next_lines(count, base, target, action):

	result = ""
	for i in range(len(base)):
		if base[i] == None:
			base[i] = 'NaN'

		if target[i] == None:
			target[i] = 'NaN'

		if base[i] == None:
			action[i] = 'NaN'

		result += str(count)+', '+base[i].strip()+', '+target[i].strip()+', '+action[i].strip()+'\n'

	return result


if __name__ == '__main__':
	
	#argv[1] -> input file, argv[2] -> output file
	if len(sys.argv) == 3:
		input_filename = str(sys.argv[1])
		output_filename = str(sys.argv[2])

		signals = utilities.read_by_line("./2018/analogy_signals.txt")

		count = 0
		sentences = TEST#utilities.readFile(input_filename)
		result = "" #goes to output file

		for next_sent in sentences:
			print(next_sent)
			base, target, action = extract_base_and_target(next_sent, signals)
			print("BASE: ", base, "TARGET: ", target, "ACTION: ", action)
			#convert the following lists into strings.
			next_lines = produce_next_lines(count, base, target, action)

			result += next_lines

			count += 1

		utilities.writeCSVFile(result, output_filename )

	else:
		print("Please specify both input and output files!")