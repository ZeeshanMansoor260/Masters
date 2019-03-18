from __future__ import division
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)
from evaluate_utils import *
import glob
import numpy as np
import pandas as pd
from multiprocessing import Pool
import contextlib
from sklearn.metrics import confusion_matrix
#from retrofit import *
import random

binary = 1
delLabel = ["vldb","icse","sigmod"]


# lite embedding container
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
class container:
    def __init__(self,labels):
        self.labels = labels

def cm2df(cm, labels):
    df = pd.DataFrame(columns=[])
    # rows
    for i, row_label in enumerate(labels):
        rowdata={}
        # columns
        for j, col_label in enumerate(labels): 
            rowdata[col_label]=cm[i,j]
        df = df.append(pd.DataFrame.from_dict({row_label:rowdata}, orient='index'))
    return df[labels]


def classify(model,data_path,classifier="lr",ratio=7,cv=10):
    #model = embeddings(filename)
    clf = ["lr"]
    result = []
    rmicro = []
    rmacro = []	
    fidx = open(data_path+'/idxs.txt')
    flabel = open(data_path+'/labels.txt')
    labels = defaultdict(list)
    labels_all = defaultdict(list)
    for idx,label in zip(fidx,flabel):
        labels[label[:-1]].append(idx[:-1])
    #for lkey in labels_all.keys():
    #    tidxs = random.sample(labels_all[lkey],1500)
    #    for tidx in tidxs:
    #        labels[lkey].append(tidx)
    print("length labels2:", len(labels))
    #print(lkey, " label length: ", len(labels[lkey]))
    fidx.close()
    flabel.close()
    for r in [i/10 for i in range(ratio,10)]:
        data = defaultdict(list)
        micro,macro, y_test, y_pred = evaluate(model,container(labels),classifier=classifier,ratio=[r],cv=cv,normalize=False,fast=True, return_y=True)
        print("r: ", r, " micro: ", np.mean(micro), " macro: ", np.mean(macro))
        result.append("r: "+ str(r)+ " micro: "+ str(np.mean(micro))+ " macro: "+ str(np.mean(micro)))
        mat = confusion_matrix(y_test,y_pred)
        rmicro.append(micro)
        rmacro.append(macro)
        print(mat)
        result.append(str(mat))
        #mat = confusion_matrix(y_test,y_pred)
        #cm_as_df=cm2df(mat,["label1","label2"])
        precision = mat[0][0] / (mat[0][0] + mat[1][0])
        recall = mat[0][0] / (mat[0][0] + mat[0][1])
        #print("precision: ", precision, " recall: ", recall)
    return result,rmicro,rmacro

