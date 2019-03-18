from P2VDataIterator import *
from gensim.models.doc2vec import TaggedDocument
import os
import logging
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
from collections import Counter
import gensim



class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list,LabeledSentence):
       self.labels_list = labels_list
       self.doc_list = doc_list
       self.LabeledSentence = LabeledSentence
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield self.LabeledSentence(words=doc.split(),tags=[self.labels_list[idx]])


def GetD2V(data_path,output_path,save_file,size=100,window=10,negative=15,min_count=10):
	LabeledSentence = gensim.models.doc2vec.LabeledSentence

#for DBLP
#data_path = "/home/mansoorz/thesis/dataCode/data/"
#For DBLPAdv
#data_path = "/home/mansoorz/Thesis2/DataSets/DBLPadv/DBLP_adv/SetData/"
#For Arxiv
#data_path = "/home/mansoorz/Thesis2/DataSets/ArXiv/6Labels/"
#data_path = "/home/mansoorz/Thesis2/DataSets/ArXiv/40k/Abstract/"
#output_path = "/home/mansoorz/Thesis2/RelatedWord/Doc2Vec/Arxiv40kAbstract/"
	data = DataIterator(data_path)
	paper = DataIterator.__iter__(data)
	data2 = list()
	labelID = list()
	for idx,title in paper:
		#print(idx + " : "+ title)
		data2.append(title)
		labelID.append(idx)

	it = LabeledLineSentence(data2, labelID,LabeledSentence)
	model = gensim.models.Doc2Vec(size=100,dm=0, window=10,hs=0,negative=15,min_count=10, workers=4)
	model.build_vocab(it)
	for epoch in range(10):
		model.train(it,epochs=model.iter,total_examples=model.corpus_count,compute_loss=True)
    #model.alpha -= 0.002 # decrease the learning rate
    #model.min_alpha = model.alpha # fix the learning rate, no deca
    #model.train(it,epochs=model.iter,total_examples=model.corpus_count)
	#model.save(output_path +"20181122doc2vec_titleAll_size100_dm0_window5_mincount5.model")
	model.save_word2vec_format(output_path +save_file,doctag_vec=True, 	prefix='', word_vec=False, binary=False)


