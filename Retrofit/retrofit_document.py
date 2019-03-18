from __future__ import division
import argparse
import gzip
import math
import numpy
import re
import sys
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn import cross_validation
from datetime import datetime


''' Write doc vectors to file '''
def print_doc_vecs(docVectors, outFileName):
	#sys.stderr.write('\nWriting down the vectors in '+outFileName+'\n')
	outFile = open(outFileName, 'w') 
	#vocabSize = len(docVectors.keys())
	vocabSize = docVectors.voc_size
	#print("Vocab Size: ", vocabSize)
	outFile.write(str(vocabSize) + " "+ str(docVectors.dimension) +"\n") 
	count = 0
	for index,paper in enumerate(docVectors.paper2id):
		outFile.write(paper + ' ')
		for val in docVectors.paper_embeddings[docVectors.paper2id[paper]]:
			outFile.write('%.4f' %(val) + ' ')
		outFile.write('\n')    
	outFile.close()
  
''' Read the PPDB doc relations as a dictionary '''
def read_lexicon(filename):
	count = 0
	lexicon = {}
	for line in open(filename, 'r'):
		docs = line.lower().rstrip().split()
		#if len(docs)==0:
		#	count +=1
		#	continue
		lexicon[docs[0]] = [doc for doc in docs[1:]]
	#print("*********Missed docs count: ",count,"********")
	return lexicon

''' Retrofit doc vectors to a lexicon '''
def retrofitOrig(docVecs, lexicon, numIters, beta):
    #count = 0
    newDocVecs = deepcopy(docVecs)
    #count = 0
    wvVocab = set(newDocVecs.paper2id.keys())
    #print("wvVocab: ",len(wvVocab))
    loopVocab = wvVocab.intersection(set(lexicon.keys()))
    print("loopVocab: ",len(loopVocab))
    for it in range(numIters):
        print("iter: ", it)
        # loop through every node also in ontology (else just use data estimate)
        for doc in loopVocab:
            count = 0
            docNeighbours = set(lexicon[doc]).intersection(wvVocab)
            #docNeighbours = set(lexicon[doc])
            #count += 1
            #if count % 10000 == 0:
            #    print(count)
            numNeighbours = len(docNeighbours)
            #no neighbours, pass - use data estimate
            #print("Num of neighbours: ", numNeighbours)
            if numNeighbours == 0:
                count += 1
                print("No neighbours: ", doc)
                continue
            # the weight of the data estimate if the number of neighbours
            # doubling the weight
            newVec = (1-beta)*numNeighbours * docVecs.paper_embeddings[docVecs.paper2id[doc]]
            # loop over neighbours and add to new vector (currently with weight 1)
            # increased weight to 2
            for ppWord in docNeighbours:
                newVec = newVec + beta*newDocVecs.paper_embeddings[docVecs.paper2id[ppWord]]
                newDocVecs.paper_embeddings[docVecs.paper2id[doc]] = newVec/(2*numNeighbours)
            #print("Missed Count: ", count)
    return newDocVecs
