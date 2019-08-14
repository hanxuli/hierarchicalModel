from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential,Model
from keras.layers import Layer,Multiply,Input, Dense, Embedding,Dropout,Activation,Bidirectional,Convolution1D, MaxPooling1D, Flatten
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import  Tokenizer
from nested_lstm import NestedLSTM
import  keras
from keras.layers.normalization import BatchNormalization
from ind_rnn import IndRNN
from keras import backend as K
import numpy as np
from keras.optimizers import Adam
from keras import initializers


num_classes=10
max_features = 100000
maxlen = 500  #2000 cut texts after this number of words (among top max_features most common words)
batch_size =128



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
y_train = keras.utils.to_categorical(y_train,num_classes=10)
embed_size=len(tokenizer.word_index)

#训练数据预处理
x_test,y_test=predataset("../data/IMDB/test.txt")
# 分词并对句子进行序列映射
x_test = tokenizer.texts_to_sequences(x_test)
y_test = keras.utils.to_categorical(y_test,num_classes=10)


#测试数据预处理
x_val,y_val=predataset("../data/IMDB/dev.txt")
# 分词并对句子进行序列映射
x_val = tokenizer.texts_to_sequences(x_val)
y_val = keras.utils.to_categorical(y_val,num_classes=10)

print('Pad sequences (samples x time)')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
x_val = sequence.pad_sequences(x_val, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('x_val shape:', x_val.shape)

# configuration matches 4.47 Million parameters with `units=600` and `64 embedding dim`
print('Build model...')


inputs=Input(shape=(maxlen,))
embed=Embedding(embed_size+1, 128, input_shape=(maxlen,))(inputs)
indRNN=Bidirectional(IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.0, recurrent_dropout=0.0,
                 return_sequences=True))(embed)
#注意力
x_score = Convolution1D(
        filters=1,
        kernel_size=3,
        padding='same',
        activation='sigmoid')(indRNN)
x_atten = Multiply()([x_score, embed])

#卷积层
cnn=Convolution1D(filters=32,
                    kernel_size=5,
                    strides=1,
                    padding='same')(x_atten)

ac=Activation('relu')(cnn)
pool=MaxPooling1D(pool_size=2,
                       strides=2,
                       padding='same')(ac)

#注意力
cnn_x_score = Convolution1D(
        filters=1,
        kernel_size=3,
        padding='same',
        activation='sigmoid')(pool)
cnn_x_atten = Multiply()([cnn_x_score, cnn])

fc=Dense(128, kernel_initializer='he_normal')(cnn_x_atten)
ac=Activation('relu')(fc)
output = Dropout(0.5)(ac)
output=Dense(num_classes, activation='softmax')(output)
model=Model(input=[inputs],output=output)


# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

model.summary()

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=3,
          shuffle=True,
          validation_data=(x_val, y_val),
          callbacks=[ModelCheckpoint('weights/IMDB_indrnn.h5', monitor='val_acc',
                                     save_best_only=True, save_weights_only=False)])

model.load_weights('weights/IMDB_indrnn.h5')

score, acc = model.evaluate(x_test, y_test,batch_size = batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
