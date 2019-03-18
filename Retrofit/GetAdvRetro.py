import numpy as np
from retrofit_document_adv import *

class embeddings:
	def __init__(self,filename,data = None):
		self.word2id = self.paper2id = {}
		self.id2word = self.id2paper = []
		with open(filename,'r') as f:
			self.voc_size,self.dimension = [int(x) for x in f.readline().rstrip().split()]
			self.word_embeddings = self.paper_embeddings = np.zeros((self.voc_size,self.dimension))
			#print("Voc size: ", self.voc_size, " dimension: ", self.dimension)
			for line in f:
				#print(line)
				line = line.split()
				if len(line)==2:
					print("Escaping: ",line)
					continue
				#print("Line Length: ", len(line))
				#print("self word2id: ", len(self.word2id))
				#print("self id2word: ", len(self.id2word))
				if len(line) <= self.dimension:
					line = [""] + line
				self.word_embeddings[len(self.id2word)] = [float(x) for x in line[1:]]
				self.word2id[line[0]] = len(self.word2id)
				self.id2word.append(line[0])
		self.data = data

def getAdvEmb(filename,lexicon,output_file,itr,beta):
	model = embeddings(filename)	
	lexicon = read_lexicon(lexicon)
	new_docvecs = retrofit(model, lexicon, itr,beta) 
	#print("saving file: ", os.path.basename(filename)[:-4]+".emb")
	print_doc_vecs(new_docvecs,output_file)
