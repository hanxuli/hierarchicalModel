from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential,Model
from keras.layers import Input,Concatenate,merge,Activation,Multiply,GlobalMaxPooling1D,Convolution1D,Dense, K,Layer, Embedding,Dropout,Bidirectional, MaxPooling1D,AveragePooling1D,LSTM
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import  Tokenizer
from keras import initializers, regularizers, constraints
from keras.callbacks import Callback
import  keras
import numpy as np
from keras.optimizers import Adam,Adadelta
import  os

num_classes=10
maxlen = 2000  #2000 cut texts after this number of words (among top max_features most common words)
batch_size =256
filter_lengths = [3,4,5] # CNN 卷积核大小
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


    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(Attention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

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
#weights=[embedding_W]
data_input=Input(shape=(maxlen,))
embed=Embedding(len(word_index)+1, 300, input_shape=(maxlen,),trainable=True,weights=[embedding_matrix])(data_input)
pool=MaxPooling1D()(embed)
dropout=Dropout(0.5)(pool)
doc_features = []
for filter_length in filter_lengths:
    conv_out = Convolution1D(
        filters=nb_filter,
        kernel_size=filter_length,
        padding='same',
        activation='tanh')(dropout)
    doc_features.append(conv_out)
doc_representation = Concatenate()(doc_features)
#pooling_out = MaxPooling1D()(conv_out)
attention=Attention()(doc_representation)
dr=Dropout(0.5)(attention)
dense=Dense(10,activation='softmax')(dr)
model=Model(inputs=data_input,outputs=dense)

# try using different optimizers and different optimizer configs
adam=Adadelta(lr=0.01)
model.compile(
        loss='categorical_crossentropy',
        optimizer=adam,
        metrics=['accuracy']
    )

model.summary()

print('Train...')

testCallback = ValTestLog(x_test=x_test, y_test=y_test,batch_size=batch_size)
model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=40,
              shuffle=True,
              validation_data=(x_val, y_val),
              callbacks=[testCallback])


