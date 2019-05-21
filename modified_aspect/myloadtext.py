# -*- coding:utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from attention_keras import Attention
import keras
from sklearn.svm import SVC
import numpy as np

import sys

reload(sys)

sys.setdefaultencoding('utf-8')



#infile=open('../acl-scl-data/depara/parallel/dvd.test.trans.out','r')
infile=open('../acl-scl-data/allen.4word2vec.review.out','r')
xtxt=[]
y=[]
for line in infile:
    a=line.strip().split(',')
    xtxt.append(a[4])
    if float(a[2])>2:
        y.append(1.0)
    else:
        y.append(0.0)
infile.close()
n_features=10000
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=3,
                                max_features=n_features,
                                stop_words='english')#,vocabulary=wdic)
xonehot=tf_vectorizer.fit_transform(xtxt)
wdic=tf_vectorizer.vocabulary_
y=np.array(y)

indices = np.arange(xonehot.shape[0])
np.random.shuffle(indices)
xonehot = xonehot[indices]
y = y[indices]

VALIDATION_SPLIT=0.9

nb_validation_samples = int(VALIDATION_SPLIT * xonehot.shape[0])


x_train = xonehot[:-nb_validation_samples]
y_train = y[:-nb_validation_samples]
x_val = xonehot[-nb_validation_samples:]
y_val = y[-nb_validation_samples:]




print len(wdic)
outdic=open('vocab.dic','w')
for i in wdic:
    outdic.write(str(i)+":"+str(wdic[i])+"\n")
outdic.close()
Xtr,sXte,sytr,syte =train_test_split(xonehot, y, test_size=0.3, random_state=0)





print('x_train shape:', Xtr.shape)
print('x_train shape:', x_train.shape)
print('x_test shape:', sXte.shape)
print('x_val shape:', x_val.shape)

#clf = SVC()
#clf.fit(x_train, y_train)
#avgacc=clf.score(x_val, y_val)
print "svm acc is "
#print avgacc

from keras.models import Model
from keras.layers import *

max_features =n_features 
batch_size = 10

S_inputs = Input(shape=(None,), dtype='int32')
embeddings = Embedding(max_features, 128)(S_inputs)
# embeddings = Position_Embedding()(embeddings) # 增加Position_Embedding能轻微提高准确率
O_seq = Attention(8,16)([embeddings,embeddings,embeddings])
O_seq = GlobalAveragePooling1D()(O_seq)
O_seq = Dropout(0.5)(O_seq)
outputs = Dense(1, activation='sigmoid')(O_seq)

model = Model(inputs=S_inputs, outputs=outputs)
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=50,
          validation_data=(sXte, syte))







#vectorizer = CountVectorizer(min_df=0, lowercase=False)
#vectorizer.fit(sentences)
#vectorizer.vocabulary_
