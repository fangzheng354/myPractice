import gensim
import codecs
import numpy as np

from gensim.models import KeyedVectors

#### read from https://radimrehurek.com/gensim/models/word2vec.html

if __name__ == '__main__':
    #model_file = '/home/nisp/fz/tmp/acl-scl-data/en-emb.bin'
    model_file = './acl-scl-data/en-emb.txt'
    #model = gensim.models.Word2Vec.load(model_file)
    wv_from_text = KeyedVectors.load_word2vec_format(model_file, binary=False)
    #print(model.wv['and'])


import keras.preprocessing.text as T
from keras.preprocessing.text import Tokenizer

text1='some thing to eat'
text2='some thing to drink'
texts=[text1,text2]

print T.text_to_word_sequence(text1)  #['some', 'thing', 'to', 'eat']
print T.one_hot(text1,10)  #[7, 9, 3, 4]
print T.one_hot(text2,10)  #[7, 9, 3, 1]

tokenizer = Tokenizer(num_words=10)
tokenzier.fit_on_text(texts)
print tokenizer.word_count #[('some', 2), ('thing', 2), ('to', 2), ('eat', 1), ('drink', 1)]
print tokenizer.word_index #{'some': 1, 'thing': 2,'to': 3 ','eat': 4, drink': 5}
print tokenizer.word_docs #{'some': 2, 'thing': 2, 'to': 2, 'drink': 1,  'eat': 1}
print tokenizer.index_docs #{1: 2, 2: 2, 3: 2, 4: 1, 5: 1}

print tokenizer.text_to_sequences(texts) #[[1, 2, 3, 4], [1, 2, 3, 5]]
print tokenizer.text_to_matrix(texts) #
[[ 0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  1.,  1.,  1.,  0.,  1.,  0.,  0.,  0.,  0.]]

import keras.preprocessing.sequence as S
S.pad_sequences([[1,2,3]],10,padding='post')
--------------------- 
作者：vivian_ll 
来源：CSDN 
原文：https://blog.csdn.net/vivian_ll/article/details/80795139 
版权声明：本文为博主原创文章，转载请附上博文链接！
