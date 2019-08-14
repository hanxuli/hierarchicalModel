#-*- coding: UTF-8 -*-  
import copy

import numpy
from keras.preprocessing.sequence import pad_sequences
from functools import reduce


def genBatch(data):
    m = 820
    maxsentencenum = len(data[0])
    for doc in data:
        for sentence in doc:
            if len(sentence) > m:
                m = len(sentence)
        for i in range(maxsentencenum - len(doc)):
            doc.append([-1])
    tmp = [numpy.asarray([sentence + [-1]*(m - len(sentence)) for sentence in doc], dtype = numpy.int32) for doc in data]
    tmp = [t+1 for t in tmp]
    return numpy.asarray(tmp)
            
def genLenBatch(lengths,maxsentencenum):
    lengths = [numpy.asarray(length + [1.0]*(maxsentencenum-len(length)), dtype = numpy.float32)+numpy.float32(1e-4) for length in lengths]
    return reduce(lambda x,y : numpy.concatenate((x,y),axis = 0),lengths)

def genwordmask(docsbatch):
    mask = copy.deepcopy(docsbatch)
    mask = [[[1.0 ,0.0][y == -1] for y in x] for x in mask]
    mask = numpy.asarray(mask,dtype=numpy.float32)
    return mask

def gensentencemask(sentencenum):
    maxnum = sentencenum[0]
    mask = numpy.asarray([[1.0]*num + [0.0]*(maxnum - num) for num in sentencenum], dtype = numpy.float32)
    return mask.T

class Dataset(object):
    def __init__(self, filename, emb,maxbatch = 32,maxword = 500):
        lines = [x.split('\t\t') for x in open(filename).readlines()]           
        label = numpy.asarray(
            [int(x[2])-1 for x in lines],
            dtype = numpy.int32
        )
        docs = [x[3][0:len(x[3])-1] for x in lines] 
        docs = [x.split('<sssss>') for x in docs] 
        docs = [[sentence.split(' ') for sentence in doc] for doc in docs]
        docs = [[[wordid for wordid in [emb.getID(word) for word in sentence] if wordid !=-1] for sentence in doc] for doc in docs]
        tmp = list(zip(docs, label))
        #random.shuffle(tmp)
        tmp.sort(lambda x, y: len(y[0]) - len(x[0]))  
        docs, label = list(zip(*tmp))

        # sentencenum = map(lambda x : len(x),docs)
        # length = map(lambda doc : map(lambda sentence : len(sentence), doc), docs)
        self.epoch = len(docs) / maxbatch
        if len(docs) % maxbatch != 0:
            self.epoch += 1
        
        self.docs = []
        self.label = []
        self.length = []

        for i in range(self.epoch):
            docsbatch = genBatch(docs[i*maxbatch:(i+1)*maxbatch])
            self.docs.append(docsbatch)
            self.label.append(numpy.asarray(label[i*maxbatch:(i+1)*maxbatch], dtype = numpy.int32))


class Wordlist(object):
    def __init__(self, filename, maxn = 100000):
        lines = [x.split() for x in open(filename,encoding="utf8").readlines()[:maxn]]
        self.size = len(lines)

        self.voc = [(item[0][0], item[1]) for item in zip(lines, range(self.size))]
        self.voc = dict(self.voc)

    def getID(self, word):
        try:
            return self.voc[word]
        except:
            return -1

def padDocs(dataset):
    for indx in range(dataset.epoch):
        docs = []
        for doc in dataset.docs[indx]:
            doc_pad = pad_sequences(doc,maxlen=130, truncating='post')
            docs.append(doc_pad)
        dataset.docs[indx] = numpy.asarray(docs)
    return dataset

# dataname = "IMDB"
# classes = 10
# voc = Wordlist('../data/'+dataname+'/wordlist.txt')
#
# print 'data loadeding....'
# trainset = Dataset('../data/'+dataname+'/test.txt', voc)
# trainset = padDocs(trainset)
# print trainset.docs[3].shape
# print trainset.docs
# f = open('../data/IMDB/testset.save','wb')
# cPickle.dump(trainset, f, protocol=cPickle.HIGHEST_PROTOCOL)
# f.close()
# print 'data load finish...'

'''
lines = map(lambda x: x.split('\t\t'), open('../data/IMDB/test.txt').readlines())
label = numpy.asarray(
    map(lambda x: int(x[2]) - 1, lines),
    dtype=numpy.int32
)
docs = map(lambda x: x[3][0:len(x[3]) - 1], lines)
docs = map(lambda x: x.split('<sssss>'), docs)
docs = map(lambda doc: map(lambda sentence: sentence.split(' '), doc), docs)
length = map(lambda doc: map(lambda sentence: len(sentence), doc), docs)
maxsentencelen = max(map(lambda doc: max(doc), length))

import nltk
fdist = nltk.FreqDist()
fdist_sent = nltk.FreqDist()
totalsentlen = 0
for doc in length:
    doclen = len(doc)
    fdist_sent[doclen] += 1
    # for senlen in doc:
    #     totalsentlen += senlen
    #     fdist[senlen] += 1

print fdist_sent.keys()
print len(fdist_sent.keys())
print sum(fdist_sent.values())
print fdist_sent.plot(74, cumulative=True)
'''
# print len(fdist.keys())
# items = sorted(fdist.items(), lambda a,b: a[1] - b[1])
# print sum(fdist.values())
# print items
# # print fdist.items()
# print maxsentencelen
# print totalsentlen //sum(fdist.values())
# fdist.plot(225, cumulative=True)


