from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.callbacks import ModelCheckpoint
from keras.datasets import imdb
from keras.preprocessing.text import  Tokenizer
from nested_lstm import NestedLSTM
from sklearn.feature_extraction.text import TfidfVectorizer
import  numpy as np
import  keras

num_classes=10
max_features = 100000
maxlen = 2000 # cut texts after this number of words (among top max_features most common words)
batch_size = 256
units=64

def predataset(file):
    train=[]
    label=[]
    max=0
    with open(file,encoding="utf8") as file:
        for line in file:
            temp=line.split("\t\t")
            train.append(temp[3])
            label.append(int(temp[2])-1)
            if max<len(temp[3]):
                max=len(temp[3]);
    print("max",max)
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

# configuration matches 4.47 Million parameters with `units=600` and `64 embedding dim`
print('Build model...')
model = Sequential()
model.add(Embedding(embed_size+1, 128, input_shape=(maxlen,)))
model.add(NestedLSTM(units, depth=2, dropout=0.0, recurrent_dropout=0.0))
model.add(Dense(10, activation='softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=6,
          validation_data=(x_val, y_val),
          callbacks=[ModelCheckpoint(('weightsimdb_nlstm_%d_%d.h5' % (units,batch_size) ),
                                     monitor='val_acc',
                                     save_best_only=True, save_weights_only=False)])

model.load_weights(('weightsimdb_nlstm_%d_%d.h5' % (units,batch_size )))

score, acc = model.evaluate(x_test, y_test,batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
