from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Flatten, Activation, Multiply, Convolution1D, RepeatVector, ZeroPadding1D, Dense, K, Layer, \
    Embedding, Dropout, Bidirectional, MaxPooling1D, AveragePooling1D, LSTM, Permute
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import  Tokenizer
from keras import initializers, regularizers, constraints
from keras.callbacks import Callback,ModelCheckpoint
import  keras
import numpy as np
from keras.models import Model
import  os
from keras.models import load_model

num_classes=10
maxlen = 2000  #2000 cut texts after this number of words (among top max_features most common words)
batch_size =128
filter_length = 3 # CNN 卷积核大小
nb_filter=100

class ValTestLog(Callback):

    def __init__(self,x_test,y_test,batch_size):
        self.val_acc = []
        self.test_acc = []
        self.x_test = x_test
        self.y_test = y_test
        self.batch_size=batch_size
    def on_epoch_end(self, epoch, logs={}):
        acc_val = logs.get('val_acc')
        self.val_acc.append(acc_val)
        score, acc_test = self.model.evaluate(self.x_test,self.y_test,batch_size=self.batch_size)
        self.test_acc.append(acc_test)
        print(("test -los:{}  -acc:{}".format(score, acc_test)))

    def on_train_end(self, logs={}):
        val_test_acc = [(val, test) for val, test in zip(self.val_acc,self.test_acc)]
        val_test_acc = sorted(val_test_acc,key=lambda a:a[1],reverse=True)
        print(("BestTestAcc:{}".format(val_test_acc[0])))
        with open("./indRnnresult", 'a') as f:
            f.write("Model bset val_acc and test_acc:\n")
            f.write(str(val_test_acc)+"\n")

def dot_product(x,kernel):

    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class Attention(Layer):
    def __init__(self, nb_filter=1, filter_length=5, padType='valid', **kwargs):
        self.nb_filter = nb_filter
        self.filter_length = filter_length
        self.padType = padType
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3, 'attention layer 输入是三维矩阵'
        self.input_dim = input_shape[2]

        self.atten_layer = Convolution1D(
            filters=self.nb_filter,
            kernel_size=self.filter_length,
            padding=self.padType,
            activation='sigmoid',
            name='attenLayer',
        )
        self.trainable_weights = self.atten_layer.trainable_weights
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        x_padding = x
        if self.padType == 'valid':
            x_padding = ZeroPadding1D(padding=self.filter_length // 2)(x)

        word_score_filters = self.atten_layer(x_padding)
        word_score_filters = Flatten()(word_score_filters)
        word_score = RepeatVector(self.input_dim)(word_score_filters)
        word_score = Permute((2, 1))(word_score)
        # out = Multiply()([x, word_score])
        return word_score

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

x_train = sequence.pad_sequences(x_train, maxlen=maxlen,padding='post',truncating='post')
x_test = sequence.pad_sequences(x_test, maxlen=maxlen,padding='post',truncating='post')
x_val = sequence.pad_sequences(x_val, maxlen=maxlen,padding='post',truncating='post')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('x_val shape:', x_val.shape)

# 读取文件，转成向量D:\pycode\hierarchicalModel
embeddings_index = {}

with open(os.path.join("../data/", 'glove.840B.300d.txt'),encoding='utf8') as f:
    for line in f:
        values = line.rstrip().split(" ")
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
#weights=[embedding_W]\

model = Sequential()
model.add(Embedding(len(word_index)+1, 300, input_shape=(maxlen,),trainable=True,weights=[embedding_matrix]))
model.add(Dropout(0.5))
#model.add(AveragePooling1D())
model.add(MaxPooling1D())
model.add(Bidirectional(LSTM(256,return_sequences=True)))
model.add(Attention(name="att"))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

# try using different optimizers and different optimizer configs

model.compile(
        loss='categorical_crossentropy',
        optimizer='adadelta',
        metrics=['accuracy']
    )

model.summary()

print('Train...')

testCallback = ValTestLog(x_test=x_test, y_test=y_test,batch_size=batch_size)
model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=20,
              shuffle=True,
              validation_data=(x_val, y_val),
              callbacks=[ModelCheckpoint('att.h5', monitor='val_acc', verbose=0,
                 save_best_only=True, save_weights_only=False)])

model.save('att.h5')

dd = "this is a stunningly beautiful movie . <sssss> the music by phillip glass is just a work of pure genius ."
origin = []
origin.append(dd)
sequences = tokenizer.texts_to_sequences(origin)
print(sequences)
# 词——下标字典
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=maxlen)


# get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
#                                       [model.layers[4].output])
#     # output in test mode = 0
# layer_output = get_3rd_layer_output([data,2])[0]
# print(layer_output)



dense1_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer('att').output)
dense1_output = dense1_layer_model.predict(data)
print (dense1_output.shape)
print ('预测值',dense1_output[0])
print(len(dense1_output[0]))

#
b=[x[0] for x in dense1_output[0]]
print(b)
new_array=[str(x) for x in b]
print(new_array)
fl=open('D:/pycode/hxlModel/data/weight.txt', 'w')
for i in new_array:
    fl.write(i)
    fl.write("\n")
fl.close()


# b=model.trainable_weights
# print('b',b)
# c=model.get_weights()[4]
# print('jsksk',c)
#
# get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
#                                       [model.layers[6].output])
# dense1_output = dense1_layer_model.predict(x_test)
#
# layer_output = get_3rd_layer_output([x_test])[0]
# print('ZUIHOU',layer_output)