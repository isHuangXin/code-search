'''
Utils for similarity computation
@author: isHuangXin
'''
import numpy as np

def cos_np(data1,data2):
    """numpy implementation of cosine similarity for matrix"""
    dotted = np.dot(data1,np.transpose(data2))
    norm1 = np.linalg.norm(data1,axis=1)
    norm2 = np.linalg.norm(data2,axis=1)
    matrix_vector_norms = np.multiply(norm1, norm2)
    neighbors = np.divide(dotted, matrix_vector_norms)
    return neighbors

def normalize(data):
    """normalize matrix by rows"""
    normalized_data = data/np.linalg.norm(data,axis=1).reshape((data.shape[0], 1))
    return normalized_data

def cos_np_for_normalized(data1,data2):
    """cosine similarity for normalized vectors"""
    return np.dot(data1,np.transpose(data2))


##### Converting / reverting #####
def convert(vocab, words):
    """convert words into indices"""
    if type(words) == str:
        words = words.strip().lower().split(' ')
    return [vocab.get(w, 0) for w in words]
def revert(vocab, indices):
    """revert indices into words"""
    ivocab = dict((v, k) for k, v in vocab.items())
    return [ivocab.get(i, 'UNK') for i in indices]

##### Padding #####
def pad(data, len=None):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)