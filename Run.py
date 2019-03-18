#***********************************************************************#
#Author: Zeeshan Mansoor						#
#Use this code to get D2V embs of the content present in the Data folder#
#d2v embs will be retrofited and classification score will be saved in	#
#the Classication_Result folder						#
#It will also visualize the embeddings and will save the images in the 	#
#Classifcation folder too						#
#									#									#
#***********************************************************************#
from GetD2VEmb import *
from Classification import *
from Retrofit.GetAdvRetro import *
from Retrofit.retrofit_document import *
from PlotData import *


#Will read embeddings into memory
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

#Classify the embeddings
def runClassify(model,name,clf,r,cv):
	with open(log_path+log_file,"a") as f:
		print("**Classifying d2v emb***")
		results,micros,macros = classify(model,data_path,classifier=clf,ratio=r,cv=cv)
		f.write("\nData_path: " + data_path + "\n")
		f.write("Emb: " + name + "\n\n")
		for result in results:
			f.write(result + "\n")
	f.close()


data_path = "Data/" #data path folder
emb_output = "Embs/" #embs will be saved in this folder
d2v_emb = "20190315_DBLP_dim100_dm0_win10_mincount10_negative15.emb" #d2v emb name
flinks = "/home/mansoorz/Thesis3/Datasets/DBLP/Undirected/UndirectedLexicon.txt"
log_path = "Classification_Result/"
log_file = "20190317_Retrofit.txt" 
beta = 0.5

#****Will get d2v embs and get classification score******
GetD2V(data_path,emb_output,d2v_emb,size=100,window=10,negative=15,min_count=10)
model = embeddings(emb_output+d2v_emb)
runClassify(model,d2v_emb,"lr",7,10)
initialize(model,data_path,log_path,"D2V_PCA.jpg",1)

links = read_lexicon(flinks) #read links/citations

#****Will do original retrofitting on d2v embs and get classification score******
orig_new_docvec = retrofitOrig(model,links,10,beta)
runClassify(orig_new_docvec,"origRetro","lr",7,10)
#Visualize the embs
initialize(orig_new_docvec,data_path,log_path,"OrigRetro_PCA.jpg",1)

#****Will do advance retrofitting on d2v embs and get classification score******
adv_new_docvec = retrofit(model,links,10,beta)
runClassify(adv_new_docvec,"advRetro","lr",7,10)
#Visualize the embs
initialize(adv_new_docvec,data_path,log_path,"AdvRetro_PCA.jpg",1)



