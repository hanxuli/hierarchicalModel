# author - Richard Liao
# Dec 26 2016
import numpy as np
import re

import os
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input
from keras.layers import Embedding, Activation, Dropout,LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from keras.callbacks import Callback
from keras import backend as K
from keras.engine.topology import Layer
from nltk import tokenize
from keras import initializers, regularizers, constraints
MAX_SENT_LENGTH = 60
MAX_SENTS = 20
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2

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

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class Attention(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

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


def get_token(filelist):
    texts = []
    for f in filelist:
        with open(f, encoding="utf8") as file:
            for line in file:
                temp = line.split("\t\t")
                text=re.sub("<sssss>", "", temp[3])
                texts.append(text)

    token = Tokenizer(num_words=MAX_NB_WORDS)
    token.fit_on_texts(texts)

    return token


def pre_data(file, token):
    reviews = []
    labels = []
    texts = []
    with open(file, encoding="utf8") as file:
        for line in file:
            temp = line.split("\t\t")
            text=re.sub("<sssss>", "", temp[3])
            texts.append(text)
            sentences = tokenize.sent_tokenize(text)
            reviews.append(sentences)
            labels.append(int(temp[2]) - 1)

    data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

    for i, sentences in enumerate(reviews):
        for j, sent in enumerate(sentences):
            if j < MAX_SENTS:
                wordTokens = text_to_word_sequence(sent)
                k = 0
                for _, word in enumerate(wordTokens):
                    if k < MAX_SENT_LENGTH and token.word_index.get(word,0) < MAX_NB_WORDS:
                        data[i, j, k] = token.word_index.get(word,0)
                        k = k + 1

    labels = to_categorical(np.asarray(labels))
    print(('Shape of data tensor:', data.shape))
    print(('Shape of label tensor:', labels.shape))

    return data, labels;


token=get_token(["../data/yelp13/train.txt","../data/yelp13/test.txt","../data/yelp13/dev.txt"])
print("token.word_index",len(token.word_index))
x_train, y_train = pre_data("../data/yelp13/train.txt",token)
x_test, y_test = pre_data("../data/yelp13/test.txt",token)
x_val, y_val = pre_data("../data/yelp13/dev.txt",token)

embeddings_index = {}
f = open(os.path.join("../data/", 'glove.6B.300d.txt'),encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print(('Total %s word vectors.' % len(embeddings_index)))
word_index=token.word_index
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in list(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in list(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True)


sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(GRU(100, return_sequences=True,recurrent_dropout=0.5))(embedded_sequences)
l_dense = TimeDistributed(Dense(200))(l_lstm)
l_att = Attention()(l_lstm)
sentEncoder = Model(sentence_input, l_att)

review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(GRU(100, return_sequences=True,recurrent_dropout=0.5))(review_encoder)
l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
l_att_sent = Attention()(l_lstm_sent)
preds = Dense(5, activation='softmax')(l_att_sent)
model = Model(review_input, preds)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
model.summary()
testCallback = ValTestLog(x_test=x_test, y_test=y_test,batch_size=64)
print("model fitting - Hierachical attention network")
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=3, batch_size=64,callbacks=[testCallback])