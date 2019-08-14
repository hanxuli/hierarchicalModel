# -*- coding: utf-8 -*-
# @auther tim
# @date 2016.11.30
# 采用双层的注意力机制,第一层针对词特征，第二层针对句子特征
# 第一层使用局部注意力机制+CNN
# 第二层可选的较多,可以使用局部注意力机制+CNN
# 或者使用 全局注意力机制+LSTM
# 或者使用 局部注意力机制+CNN
import pickle as cp

import numpy as np

from Dataset import *
from keras.callbacks import ModelCheckpoint, Callback
from keras.engine import Model, Layer
from keras.layers import Embedding, Input, Convolution1D, GlobalMaxPooling1D,Concatenate, TimeDistributed,Dense, Multiply, K, GRU, Bidirectional, Permute, Activation, Lambda
from keras.utils import to_categorical

# 参数
EMBEDDING_DIM = 200 # 词向量维度
filter_lengths = [1,3,4,5] # CNN 卷积核大小
nb_filter = 100  # 卷积神经网络积极核函数的个数
fc_hidden_dims = 200 # 全连接层隐藏单元数量
batch_size = 32
MAX_SENT_LENGTH = 130


# dataname = sys.argv[1]
# classes = sys.argv[2]
dataname = "IMDB"
classes = 10
voc = Wordlist('../data/'+dataname+'/wordlist.txt')

print('data loadeding....')
# trainset = Dataset('../data/'+dataname+'/train.txt', voc)
f = open('../data/'+dataname+'/trainset.save', 'rb')
trainset = cp.load(f,encoding='iso-8859-1')
f = open('../data/'+dataname+'/devset.save', 'rb')
devset = cp.load(f,encoding='iso-8859-1')
f = open('../data/'+dataname+'/testset.save', 'rb')
f = open('../data/'+dataname+'/testset.save', 'rb')
testset = cp.load(f,encoding='iso-8859-1')
f.close()

# for indx in range(trainset.epoch):
#     docs = []
#     for doc in trainset.docs[indx]:
#         doc_pad = pad_sequences(doc,maxlen=130, truncating='post')
#         docs.append(doc_pad)
#     trainset.docs[indx] = np.asarray(docs)

print(trainset.docs[1].shape)
print(trainset.label[1].shape)
print(trainset.epoch)
print(devset.epoch)
print(testset.epoch)

# 将数据padding成120维
# pad_sequences
print('data load finish...')

print('word embedding matrix loading')
f = open('../data/'+dataname+'/embinit.save', 'rb')
embedding_matrix = cp.load(f,encoding='iso-8859-1')
f.close()
embedding_W = np.zeros_like(embedding_matrix)
embedding_W[0] = embedding_matrix[-1]
embedding_W[1:] = embedding_matrix[0:-1]
print((embedding_W.shape))
print('word embedding matrix loading finish')

def Sum(x):
    return K.sum(x, axis=1)

def sum_output_shape(input_shape):
    return (input_shape[0], input_shape[-1])

class AttenLayer(Layer):
    def __init__(self, **kwargs):
        super(AttenLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert  len(input_shape)==3
        self.dense = Dense(1)
        self.trainable_weights = self.dense.trainable_weights
        super(AttenLayer, self).build(input_shape)

    def call(self, x, mask=None):
        atten = self.dense(x)
        atten = Permute((2, 1))(atten)
        atten = Activation('softmax')(atten)
        atten = Permute((2, 1))(atten)
        conv_feature = Multiply()([x, atten])
        conv_feature = Lambda(Sum, output_shape=sum_output_shape)(conv_feature)
        return conv_feature

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def atten(type='global', inputs=None, atten_range=None):
    assert type in ['global','local'], 'type in [global,local]'

    if type=='local' and atten_range==None:
        raise Exception('type is local, atten_range must be not None')

    # =============lstm 全局attention机制====================
    if type=='global' and inputs!=None:
        x_atten = AttenLayer()(inputs)
        return x_atten

    # =============局部attention机制====================
    elif type=='local' and inputs!=None:
        x_score = Convolution1D(
            filters=1,
            kernel_size=atten_range,
            padding='same',
            activation='sigmoid'
        )(inputs)
        x_atten = Multiply()([x_score, inputs])
        return x_atten

def lstm_atten(sent_sequences):
    sent_sequences = Bidirectional(GRU(200, return_sequences=True))(sent_sequences)

    doc_atten = atten(type='global', inputs=sent_sequences)
    return doc_atten



def conv_atten(sent_sequences,filter_lengths=[1,3,4,5],nb_filter=100,atten_range=5):

    sent_sequences = atten(type='local', inputs=sent_sequences, atten_range=atten_range)

    # 卷积神经网络提取特征
    doc_features = []
    for filter_length in filter_lengths:
        conv_out = Convolution1D(
            filters=nb_filter,
            kernel_size=filter_length,
            padding='same',
            activation='relu'
        )(sent_sequences)
        pooling_out = GlobalMaxPooling1D()(conv_out)
        # doc_conv = Dropout(0.5)(pooling_out)
        # doc_features.append(doc_conv)
        doc_features.append(pooling_out)


    doc_representation = Concatenate()(doc_features)
    return doc_representation



#================sentence========================
sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
# word embedding 层
embedding_layer = Embedding(embedding_W.shape[0],
                            EMBEDDING_DIM,
                            weights=[embedding_W],
                            trainable=True)
embedding_sequences = embedding_layer(sentence_input)

# sents_representation=lstm_atten(embedding_sequences)
sents_representation = conv_atten(embedding_sequences,filter_lengths=filter_lengths, nb_filter=nb_filter)
sent_model = Model(sentence_input, sents_representation,name='sentModel')


#==============================document================
doc_input = Input(shape=(None,MAX_SENT_LENGTH), dtype='int32')
sent_sequences = TimeDistributed(sent_model)(doc_input)

doc_representation=lstm_atten(sent_sequences)
# doc_representation = conv_atten(sent_sequences,atten_range=3)

fc_out = Dense(units=fc_hidden_dims,activation='relu',
                   name='fcLayer')(doc_representation)
preds = Dense(classes, activation='softmax')(fc_out)
model = Model(doc_input, preds)
model.compile(
        loss='categorical_crossentropy',
        optimizer='adadelta',
        metrics=['accuracy']
    )

def get_batch(dataset):
    epoch = dataset.epoch
    epochs = np.arange(epoch)

    while True:
        np.random.shuffle(epochs)
        for i in epochs:
            inpt_data = dataset.docs[i]
            outpt_data = to_categorical(dataset.label[i],num_classes=classes)
            yield (inpt_data, outpt_data)

class TestHistory(Callback):
    def __init__(self,dataset):
        self.best_acc = []
        self.dataset = dataset
    def on_epoch_end(self, epoch, logs={}):
        score, acc = self.model.evaluate_generator(get_batch(self.dataset),steps=self.dataset.epoch)
        self.best_acc.append(acc)
        print(("best test -los:{}  -acc:{}".format(score,acc)))
    def on_train_end(self,logs={}):
        print(("test acc list: "+str(self.best_acc)))
        print(("BestTest  acc:{}".format(max(self.best_acc))))


print(model.summary())
# plot_model(model, to_file='../figure/hierarchicalCNN_No_Shape.png', show_shapes=False)
# plot_model(sent_model, to_file='../figure/sentModel_No_Shape.png', show_shapes=False)
checkpointer = ModelCheckpoint(
    filepath="../save/weights.hdf5",
    verbose=1,
    monitor='val_acc',
    save_best_only=True,
    save_weights_only=False
)
testCallback = TestHistory(dataset=testset)

model.fit_generator(
    get_batch(dataset=trainset),
    steps_per_epoch=trainset.epoch,
    epochs=15,
    validation_data=get_batch(dataset=devset),
    validation_steps=devset.epoch,
    callbacks=[testCallback]
)


