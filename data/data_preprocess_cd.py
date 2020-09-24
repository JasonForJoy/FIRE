import os
import re
import random
import numpy as np
from nltk.tokenize import WordPunctTokenizer


def tokenize(text):
	return WordPunctTokenizer().tokenize(text)


RETOK = re.compile(r'\w+|[^\w\s]|\n', re.UNICODE)

def re_tokenize(text):
    """Find boundaries between word characters, newlines, and non-word
    non-whitespace tokens ``(r'[\\w\\n]+ | [^\\w\\s] | \\n')``.
    This splits along whitespace and punctuation and keeps the newline as
    a token in the returned list.
    """
    return RETOK.findall(text)


def data_process(input_path, output_path, fname):

	dialogues = []
	dialogue = []
	with open(os.path.join(input_path, fname), "r") as f:
		for line in f:
			# line = line.decode('utf-8').strip()
			line = line.strip()
			if line.split()[0] == "1":  # new dialogue
				dialogues.append(dialogue)
				dialogue = []
			dialogue.append(line)

		dialogues.append(dialogue)
		dialogues.remove([])
	print("{} is composed of {} dialogues".format(fname, len(dialogues)))

	context_candidates = []
	for dialogue in dialogues:
		persona = []
		context_history = []
		for line in dialogue:
			fields = line.strip().split("\t")

			if len(fields) == 1:
				profile = fields[0].split(" your persona: ")[-1]
				profile = " ".join(re_tokenize(profile))
				persona.append(profile)
			if len(fields) == 4:
				context = " ".join(re_tokenize(fields[0])[1:]).replace('|', '').strip()
				response = " ".join(re_tokenize(fields[1])).replace('|', '').strip()
				candidates = [" ".join(re_tokenize(c)) for c in fields[-1].split('|')]
				if "" in candidates:
					candidates.remove("")
				candidates = candidates[-20:]
				random.shuffle(candidates)
				label = candidates.index(response)

				context_history.append(context)
				# (context, candidates, label, your persona)
				context_candidates.append( [" _eos_ ".join(context_history) + " _eos_", "|".join(candidates), str(label), "|".join(persona)] )
				context_history.append(response)
	print("{} is composed of {} context-candidates".format(fname, len(context_candidates)))

	with open(os.path.join(output_path, "processed_{}".format(fname)), "w") as f:
		print("Saving dataset to processed_{} ...".format(fname))
		for dialogue in context_candidates:
			# f.write(("\t".join(dialogue) + "\n").encode('utf-8'))
			f.write(("\t".join(dialogue) + "\n"))


if __name__ == '__main__':

	input_path = "./cmudog"
	output_path = "./cmudog_processed"

	if not os.path.exists(output_path):
		os.makedirs(output_path)

	files = [file for file in os.listdir(input_path)]

	print("There are {} files to process.\nStart processing data ...".format(len(files)))

	for file in files:
		print("Preprocessing {} ...".format(file))
		data_process(input_path, output_path, file)
		print("="*60)
	
	print("data preprocess done!")
