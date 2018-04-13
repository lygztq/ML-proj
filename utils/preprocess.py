from load_data import read_dirs
import class_name
import numpy as np

def seq2matrix(seq):
    """
    Change a RNA sequence(len=300) to a one-hot matrix representation.
    e.g.
        if A=0, C=1, G=2, T=3
        ACG --> [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
        :param seq: A RNA sequence
        :return:    a one-hot matrix with size (300 * 4)
    """
    mat = np.zeros([len(seq), 4])
    for i in range(len(seq)):
        if seq[i] == 'A':
            mat[i][0] = 1
        elif seq[i] == 'C':
            mat[i][1] = 1
        elif seq[i] == 'G':
            mat[i][2] = 1
        else:
            mat[i][3] = 1
    return mat

def preprocess():
    """
    Preprocessing the dataset, change the char-RNA seq into one-hot matrix.
    
    
    Combining all the 37 classes using 0-1 vector to represent the label and
    merge the duplicated seq. 
    
    The preprocessed data:
        one_hot_data:   a (num_of_seq, 300, 4) numpy array that contains num_of_seq
                        one-hot matrix
        labels:         a (num_of_seq, 37) numpy array that contains the labels
    Save the preprocessed data into /data/datas.npy and /data/labels.npy
    """
    data, label = read_dirs('../data')
    num = data.shape[0]
    one_hot_data = np.zeros([num, len(data[0]), 4])
    for i in range(num):
        print 'processing: %d / %d' % (i+1, num)
        one_hot_data[i] = seq2matrix(data[i])
    np.save('../data/datas',one_hot_data)
    np.save('../data/labels',label)

if __name__ == '__main__':
    preprocess()