import logging
from collections import defaultdict, namedtuple
#import nltk
from glob import glob
import sys
import gzip
if sys.version_info[0] >= 3:
    unicode = str
import pickle

class DataIterator():
    def __init__(self,data_path='./'):
        self.path = data_path
        self.labels = defaultdict(list)

        fidx = open(self.path+"idxs.txt")
        flabels = open(self.path+"labels.txt")
        for idx,labels in zip(fidx,flabels):
            for label in labels.rstrip().split(" "):
                self.labels[label].append(idx)

    def __iter__(self):
        Data = namedtuple('paper', 'idx words')
        fidx = open(self.path+"idxs.txt")
        fwords = open(self.path+"words.txt")
        #fcitation = open(self.path+"citations.txt")
        for idx,words in zip(fidx,fwords):
            yield Data(
                        idx[:-1],
                        words[:-1],
                        #citation[:-1].split(" ")
                    )
