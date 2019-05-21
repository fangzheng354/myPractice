#-*- coding:utf-8 -*-
from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
#from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import nltk  #用来分词
import collections  #用来统计词频
import numpy as np
import jieba

from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Input
#from keras.layers import Dense, Dropout, Activation, Embedding, Input
from my_layers import Attention, Average, WeightedSum
from keras.models import Model



maxlen = 0  #句子最大长度
word_freqs = collections.Counter()  #词频
num_recs = 0 # 样本数
with open('./training-zh.txt','r+') as f:
    for line in f:
        label, sentence = line.strip().split(",")
        words = sentence.lower().split()
        #words = nltk.word_tokenize(sentence.lower())
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            word_freqs[word] += 1
        num_recs += 1
print('max_len ',maxlen)
print('nb_words ', len(word_freqs))

MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 40


vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word2index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v:k for k, v in word2index.items()}


X = np.empty(num_recs,dtype=list)
y = np.zeros(num_recs)
i=0
with open('./training-zh.txt','r+') as f:
    for line in f:
        label, sentence = line.strip().split(",")
        words = sentence.lower().split()
        #words = nltk.word_tokenize(sentence.lower())
        seqs = []
        for word in words:
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])
        X[i] = seqs
        y[i] = int(label)
        i += 1
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

##### train

EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64

BATCH_SIZE = 32
NUM_EPOCHS = 10

'''
model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE,input_length=MAX_SENTENCE_LENGTH))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
'''
overal_maxlen=40

MAX_SENTENCE_LENGTH = 40

dropout=0.05
recurrent_dropout=0.05
sent_input=Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32', name='sentence_input')
word_emb = Embedding(vocab_size, EMBEDDING_SIZE, mask_zero=True, name='word_emb')
sent_term_embs = word_emb(sent_input)
lstmfun = LSTM(HIDDEN_LAYER_SIZE, return_sequences=False, dropout=dropout, recurrent_dropout=recurrent_dropout, name='lstm')
sentence_output = lstmfun(sent_term_embs)
#single_output = Average(mask_zero=True)(sentence_output)
single_outD = Dense(1, name='dense_1')(sentence_output)
#single_probs = Activation('softmax', name='active_model')(single_outD)
single_probs = Activation('sigmoid', name='active_model')(single_outD)
model = Model(inputs=[sent_input], outputs=[single_probs])

''' ref code 
    sentence_input = Input(shape=(overal_maxlen,), dtype='int32', name='sentence_input')
    aspect_input = Input(shape=(maxlen_aspect,), dtype='int32', name='aspect_input')
    pretrain_input = Input(shape=(None,), dtype='int32', name='pretrain_input')

    ##### construct word embedding layer #####
    word_emb = Embedding(vocab_size, args.emb_dim, mask_zero=True, name='word_emb')

    ### represent aspect as averaged word embedding ###
    print 'use average term embs as aspect embedding'
    aspect_term_embs = word_emb(aspect_input)
    aspect_embs = Average(mask_zero=True, name='aspect_emb')(aspect_term_embs)

    ### sentence representation ###
    sentence_output = word_emb(sentence_input)
    pretrain_output = word_emb(pretrain_input)


    print 'use a rnn layer'
    rnn = RNN(args.rnn_dim, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout, name='lstm')
    sentence_output = rnn(sentence_output)
    pretrain_output = rnn(pretrain_output)

    print 'use content attention to get term weights'
    att_weights = Attention(name='att_weights')([sentence_output, aspect_embs])
    sentence_output = WeightedSum()([sentence_output, att_weights])

    print 'use content attention to get term weights'
    att_weights = Attention(name='att_weights')([sentence_output, aspect_embs])
    sentence_output = WeightedSum()([sentence_output, att_weights])

    pretrain_output = Average(mask_zero=True)(pretrain_output)

    if args.dropout_prob > 0:
        print 'use dropout layer'
        sentence_output = Dropout(args.dropout_prob)(sentence_output)
        pretrain_output = Dropout(args.dropout_prob)(pretrain_output)


    sentence_output = Dense(num_outputs, name='dense_1')(sentence_output)
    pretrain_output = Dense(num_outputs, name='dense_2')(pretrain_output)

    aspect_probs = Activation('softmax', name='aspect_model')(sentence_output)
    doc_probs = Activation('softmax', name='pretrain_model')(pretrain_output)

    model = Model(inputs=[sentence_input, aspect_input, pretrain_input], outputs=[aspect_probs, doc_probs])
# end of ref code 
''' 


model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])

model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,validation_data=(Xtest, ytest))


score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
print('{}   {}      {}'.format('预测','真实','句子'))
for i in range(5):
    idx = np.random.randint(len(Xtest))
    xtest = Xtest[idx].reshape(1,40)
    ylabel = ytest[idx]
    ypred = model.predict(xtest)[0][0]
    sent = " ".join([index2word[x] for x in xtest[0] if x != 0])
    print(' {}      {}     {}'.format(int(round(ypred)), int(ylabel), sent))


INPUT_SENTENCES = ['I love reading.','You are so boring.']
XX = np.empty(len(INPUT_SENTENCES),dtype=list)
i=0
for sentence in  INPUT_SENTENCES:
    words = nltk.word_tokenize(sentence.lower())
    seq = []
    for word in words:
        if word in word2index:
            seq.append(word2index[word])
        else:
            seq.append(word2index['UNK'])
    XX[i] = seq
    i+=1

XX = sequence.pad_sequences(XX, maxlen=MAX_SENTENCE_LENGTH)
labels = [int(round(x[0])) for x in model.predict(XX) ]
label2word = {1:'积极', 0:'消极'}
for i in range(len(INPUT_SENTENCES)):
    print('{}   {}'.format(label2word[labels[i]], INPUT_SENTENCES[i]))
