from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM,LeakyReLU,K,Layer,TimeDistributed, Multiply,Permute,Dense, Embedding,Dropout,Activation,Bidirectional,Convolution1D, MaxPooling1D, Flatten
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import  Tokenizer
from nested_lstm import NestedLSTM
import  keras
from keras.layers.normalization import BatchNormalization
from ind_rnn import IndRNN
import numpy as np
# from keras.optimizers import Adam
# import pickle as cp
# import  re
# from  GroupNormalization import  GroupNormalization
# from sru import SRU
# from highway import Highway
import  os

num_classes=10
max_features = 100000
maxlen = 2000  #2000 cut texts after this number of words (among top max_features most common words)
batch_size =64


class LAttenLayer(Layer):
    def __init__(self,nb_filter=1,filter_length=5,**kwargs):
        self.nb_filter = nb_filter
        self.filter_length = filter_length
        super(LAttenLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert  len(input_shape)==3, 'attention layer 输入是三维矩阵'
        self.input_dim = input_shape[2]

        self.atten_layer = Convolution1D(
            filters=self.nb_filter,
            kernel_size=self.filter_length,
            padding='same',
            activation='sigmoid'
        )
        self.trainable_weights = self.atten_layer.trainable_weights
        super(LAttenLayer, self).build(input_shape)

    def call(self, x, mask=None):
        x_padding = x
        word_score = self.atten_layer(x_padding)
        out = Multiply()([x, word_score])
        return out

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)

class Sum(Layer):
    def call(self, inputs, **kwargs):
        return K.sum(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

class GAttenLayer(Layer):
    def __init__(self, **kwargs):
        super(GAttenLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert  len(input_shape)==3
        self.dense = Dense(1)
        self.trainable_weights = self.dense.trainable_weights
        super(GAttenLayer, self).build(input_shape)

    def call(self, x, mask=None):
        atten = self.dense(x)
        atten = Permute((2, 1))(atten)
        atten = Activation('softmax')(atten)
        atten = Permute((2, 1))(atten) #[n_sample, n_length, 1]
        conv_feature = Multiply()([x, atten]) #[n_sample, n_length, embedding]
        conv_feature = Sum()(conv_feature) #[n_sample, embedding]
        return conv_feature

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def predataset(file):
    train=[]
    label=[]
    max_len=0
    with open(file,encoding="utf8") as file:
        for line in file:
            temp=line.split("\t\t")
            train.append(temp[3])
            label.append(int(temp[2])-1)
            if max_len<len(temp[3]):
                max_len=len(temp[3])
    print("max",max_len)
    return  train ,label

#训练数据预处理
x_train,y_train=predataset("../data/IMDB/train.txt")

# 对句子进行序列映射
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
y_train = keras.utils.to_categorical(y_train,num_classes=num_classes)
word_index = tokenizer.word_index

#测试数据预处理
x_test,y_test=predataset("../data/IMDB/test.txt")
# 分词并对句子进行序列映射
x_test = tokenizer.texts_to_sequences(x_test)
y_test = keras.utils.to_categorical(y_test,num_classes=num_classes)


#测试数据预处理
x_val,y_val=predataset("../data/IMDB/dev.txt")
# 分词并对句子进行序列映射
x_val = tokenizer.texts_to_sequences(x_val)
y_val = keras.utils.to_categorical(y_val,num_classes=num_classes)

print('Pad sequences (samples x time)')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
x_val = sequence.pad_sequences(x_val, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('x_val shape:', x_val.shape)

# 读取文件，转成向量D:\pycode\hierarchicalModel
embeddings_index = {}
with open(os.path.join("D:/pycode/hierarchicalModel/data/", 'glove.6B.300d.txt'),encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print(len(embeddings_index))
print('Preparing embedding matrix.')

embedding_matrix = np.zeros((len(word_index)+1, 300))

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
# configuration matches 4.47 Million parameters with `units=600` and `64 embedding dim`
print('Build model...')
#weights=[embedding_W]
model = Sequential()
model.add(Embedding(len(word_index)+1, 300, input_shape=(maxlen,),trainable=False,weights=[embedding_matrix]))
model.add(Dropout(0.5))
model.add(LAttenLayer())
model.add(Convolution1D(filters=32,
                        kernel_size=5,
                        strides=1,
                        padding='same'))

model.add(Activation('relu'))

model.add(MaxPooling1D(pool_size=2,strides=2,padding='same'))

model.add(Bidirectional(IndRNN(300, return_sequences=True)))
model.add(GAttenLayer())
model.add(Activation('relu'))
model.add(Dense(num_classes, activation='softmax'))

# try using different optimizers and different optimizer configs

model.compile(
        loss='categorical_crossentropy',
        optimizer='adadelta',
        metrics=['accuracy']
    )

model.summary()

print('Train...')

model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=10,
              shuffle=True,
              validation_data=(x_val, y_val),
              callbacks=[ModelCheckpoint('weights/IMDB_indrnn.h5', monitor='val_acc',
                                         save_best_only=True, save_weights_only=False)])

model.load_weights('weights/IMDB_indrnn.h5')

score, acc = model.evaluate(x_test, y_test,batch_size = batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
