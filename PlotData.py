
from __future__ import division
#from evaluate_utils import *
from evaluate_utilsPlot import *
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from collections import Counter

class embeddings:
    def __init__(self,filename,data = None):
        self.word2id = self.paper2id = {}
        self.id2word = self.id2paper = []
        with open(filename,'r') as f:
            self.voc_size,self.dimension = [int(x) for x in f.readline().rstrip().split()]
            self.word_embeddings = self.paper_embeddings = np.zeros((self.voc_size,self.dimension))
            print("Voc size: ", self.voc_size, " dimension: ", self.dimension)
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
                self.word2id[line[0].strip()] = len(self.word2id)
                self.id2word.append(line[0].strip())
        self.data = data

def tsne_plot(new_values,Y,idxs,micro,save_path,save_file):
	x = []
	y = []	
	for value in new_values:
		x.append(value[0])
		y.append(value[1])
	plt.figure(figsize=(16, 16)) 
	labels = Counter(Y).keys()
	cmap = plt.cm.get_cmap("hsv", len(labels)+1)
	colors = ["red", "blue","green"]
	for i in range(len(x)):
		plt.scatter(x[i],y[i], color=cmap(labels.index(Y[i])))
		#if Y[i] == "vldb":
		#	plt.scatter(x[i],y[i], color=colors[0])
		#elif Y[i] == "icse":
		#	plt.scatter(x[i],y[i], color=colors[1])
			#this part will add the paper id			
			#if ( i % 100) == 0:
			#	plt.text(x[i]+(i/1000), y[i]+(i/1000), idxs[i], fontsize=9)
		#else:
		#	plt.scatter(x[i],y[i], color=colors[2])
	matches = list()
	for index,label in enumerate(labels):
		#print("label: ", index, ":" ,label)		
		matches.append(mpatches.Patch(color=cmap(index), label=label))
	plt.legend(handles=matches,fontsize=32)
	plt.title('micro: %.4f' % micro,fontsize=32)
	#red_patch = mpatches.Patch(color='red', label='vldb')
	#blue_patch = mpatches.Patch(color='blue', label='icse')
	#green_patch = mpatches.Patch(color='green', label='sigmod')
	#plt.legend(handles=[red_patch,blue_patch,green_patch])
	plt.savefig(save_path + save_file)
	#plt.show()

def ClassifyPlot(new_values,Y):
	micro,macro, y_test, y_pred = evaluate(new_values,Y,classifier="lr",ratio=[0.9],cv=10,normalize=False,fast=True, return_y=True)
	print(" micro: ", micro, " macro: ", macro)
	return micro

def initialize(model,data_path,save_path,save_file,pca=0):
	#model = embeddings(filename)
	fidx = open(data_path+'/idxs.txt')
	flabel = open(data_path+'/labels.txt')
	labels = defaultdict(list)
	for idx,label in zip(fidx,flabel):
		labels[label[:-1]].append(idx[:-1])
	X = []
	Y = []
	idxs = []
	for y,key in enumerate(labels.keys()):
		for index,paper in enumerate(labels[key]):
			if paper not in model.paper2id:
				continue
			if ((index+1) % 3000) == 0:
				print(key," : ", index)			
				break
			X.append(model.paper_embeddings[model.paper2id[paper]])
			Y.append(key)
			idxs.append(paper)
	print("X: ",len(X)," y: ", len(Y), " id: ", len(idxs))
	print("id: ", idxs[3], " Label: ", Y[3])
	print("Reducing Vectors...")
	if (pca == 1):	
		pca = PCA(n_components=2)
		new_values = pca.fit_transform(X)
	else:
		tsne_model = TSNE(perplexity=15, n_components=2)
		new_values = tsne_model.fit_transform(X)
	print("Classifiying")	
	micro = ClassifyPlot(new_values,Y)
	print("new_values shape: ", np.shape(new_values))
	print("ploting data..")
	tsne_plot(new_values,Y,idxs,micro,save_path,save_file)
