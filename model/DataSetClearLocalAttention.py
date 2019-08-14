from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential,Model
from keras.layers import Dense, Embedding,Convolution1D,Input,Multiply,Dropout
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.preprocessing.text import  Tokenizer
from nested_lstm import NestedLSTM
import  keras
from ind_rnn import IndRNN
import logging
import  tensorflow as tf
from keras import optimizers
from nltk.corpus import stopwords
import re
import jieba

def Args():
    # 参数
    flags = tf.flags
    flags.DEFINE_string("dataName", "IMDB", "dataName IMDB yelp13 yelp15")  # 训练集的名字
    flags.DEFINE_string("train", "../data/IMDB/train.txt", "PATH_TO_TRAIN")  # 训练集的路径
    flags.DEFINE_string("test", "../data/IMDB/test.txt", "PATH_TO_TEST")  # 测试集的路径
    flags.DEFINE_string("val", "../data/IMDB/dev.txt", "PATH_TO_val")  # 测试集的路径
    flags.DEFINE_integer("batch_size", 256, "batch_size 32 64 128 256 512 1024")  # 训练批次的大小
    flags.DEFINE_integer("epochs",5, "epochs")  # 训练批次
    flags.DEFINE_integer("units", 128, "units 32 64 128 256 512 1024")  # 神经单元的个数
    flags.DEFINE_integer("num_classes", 10, "num_classes 5 10")  # 类别的个数
    flags.DEFINE_string("optimizer", "Adam", "optimizer")  # 优化函数
    flags.DEFINE_float("dropout", 0.1, "dropout")
    flags.DEFINE_integer("maxlen", 2000, "num_classes") # number of words
    flags.DEFINE_string("Name", "LocalAttention_indRnn", "Name")  # 程序名字
    FLAGS = flags.FLAGS

    return FLAGS


#去除停用词和<sssss>
def predataset(file):
    train=[]
    label=[]
    max_len=0
    # 去除标点符号
    english_punctuations = ['``',',', '\'','.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    with open(file,encoding="utf8") as file:
        for line in file:
            temp=line.split("\t\t")
            #去除<sssss>
            s = re.sub("<sssss>", "", temp[3])
            #分词
            s=jieba.cut(s)
            final = ''
            #去除停用词和标点符号
            for seg in s:
             if seg not in stopwords.words('english'):
                if seg not in english_punctuations:
                        final += seg
            print(final)
            train.append(final)
            label.append(int(temp[2])-1)
            if max_len<len(final):
                max_len=len(final)
    print("max",max_len)
    return  train ,label
#日志模块
# 指定logger输出格式

def log():
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("../log/"+FLAGS.dataName+".log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger



if __name__=="__main__":
    FLAGS=Args()
    logger=log()
    #logger.info('dataName:%3s,epochs:%3s,units:%6,lr:%6s,optimizer:%5s',FLAGS.dataName, FLAGS.epochs, FLAGS.units, FLAGS.lr , FLAGS.optimizer)
    logger.info('Name:%3s,dataName:%3s, batch_size:%3s, units:%s, optimizer:%5s,epochs:%3s,dropout:%3s',
                FLAGS.Name,FLAGS.dataName, FLAGS.batch_size,FLAGS.units, FLAGS.optimizer,FLAGS.epochs,FLAGS.dropout)

    num_classes=FLAGS.num_classes
    max_features = 80000
    maxlen = FLAGS.maxlen  # cut texts after this number of words (among top max_features most common words)
    batch_size = FLAGS.batch_size
    #训练数据预处理
    x_train,y_train=predataset(FLAGS.train)

    # 对句子进行序列映射
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_train)
    x_train = tokenizer.texts_to_sequences(x_train)
    y_train = keras.utils.to_categorical(y_train,num_classes=num_classes)
    embed_size=len(tokenizer.word_index)
    #训练数据预处理
    x_test,y_test=predataset(FLAGS.test)
    # 分词并对句子进行序列映射
    x_test = tokenizer.texts_to_sequences(x_test)
    y_test = keras.utils.to_categorical(y_test,num_classes=num_classes)


    #测试数据预处理
    x_val,y_val=predataset(FLAGS.val)
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

    # configuration matches 4.47 Million parameters with `units=600` and `64 embedding dim`
    print('Build model...')

    inputs=Input(shape=(maxlen,))
    embed=Embedding(embed_size+1, 128, input_shape=(maxlen,))(inputs)
    x_score = Convolution1D(
        filters=1,
        kernel_size=3,
        padding='same',
        activation='sigmoid'
    )(embed)
    x_atten = Multiply()([x_score, embed])
    first_ind=IndRNN(FLAGS.units, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.0, recurrent_dropout=0.0,
                      return_sequences=True)(x_atten)
    second_ind=IndRNN(FLAGS.units, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.0, recurrent_dropout=0.0,
                      return_sequences=False)(first_ind)

    output=Dense(num_classes, activation='softmax')(second_ind)
    output=Dropout(FLAGS.dropout)(output)
    model=Model(input=[inputs],output=output)
    # try using different optimizers and different optimizer configs

    optimizer = optimizers.SGD()
    if FLAGS.optimizer=="SGD":
        optimizer= optimizers.SGD()
    elif FLAGS.optimizer=="Adam":
        optimizer = optimizers.Adam()
    elif  FLAGS.optimizer=="Adadelta":
        optimizer = optimizers.Adadelta()
    elif  FLAGS.optimizer=="Adagrad":
        optimizer = optimizers.Adagrad()
    elif  FLAGS.optimizer=="RMSprop":
        optimizer = optimizers.RMSprop()


    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.summary()

    print('Train...')
    modelCheckpoint=ModelCheckpoint('weights/%s_%s_%s_%s.h5' % (FLAGS.dataName, FLAGS.units, FLAGS.batch_size, FLAGS.optimizer),
                    monitor='val_acc', save_best_only=True, save_weights_only=False)
    early_stop = EarlyStopping('val_loss', patience=5)
    model.fit(x_train, y_train,
              batch_size=FLAGS.batch_size,
              epochs=FLAGS.epochs,
              validation_data=(x_val, y_val),
              callbacks=[modelCheckpoint,early_stop])

    model.load_weights('weights/%s_%s_%s_%s.h5'%(FLAGS.dataName,FLAGS.units,FLAGS.batch_size,FLAGS.optimizer))

    score, acc = model.evaluate(x_test, y_test,batch_size = FLAGS.batch_size)
    logger.info('Test score:%s,Test accuracy:%s',score,acc)
    print('Test score:', score)
    print('Test accuracy:', acc)
