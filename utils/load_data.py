import numpy as np
import class_name
import os
#import matplotlib.pyplot as plt

# The part that read data of different class individually

def read_file(file_path):
    """
    Read data from a file
    
    :param file_path: The path of the file
    
    :Return : two numpy array contain the data and the label 
    """
    contain = open(file_path)
    data = []
    label = []

    for line in contain:
        line = line.split()
        data.append(line[0])
        label.append(int(line[1]))

    data = np.array(data)
    label = np.array(label)
    return data, label

# The part that read data totally
def read_dirs(dir_path):
    """
    Read data from the dataset dir, merge 
    duplicated data with different label.
    :param dir_path: The path that contains the whole dataset.
    """ 
    names = class_name.CLASS_NAMES
    indices = class_name.CLASS_INDICES
    dataset = {}

    for n in names:
        train_file = os.path.join(dir_path, n, 'train')#.replace('\\','/')
        contain = open(train_file)
        for l in contain:
            l = l.split()
            if dataset.has_key(l[0]):
                dataset[l[0]][indices[n]] = 1
            else:
                dataset[l[0]] = class_name.CLASS_NUM * [0]
    data = np.array(list(dataset.keys()))
    label = np.array(list(dataset.values()))
    return data, label

def load_data(path='../data'):
    data_path = os.path.join(path, 'datas.npy')
    label_path = os.path.join(path, 'labels.npy')

    data = np.load(data_path)
    label = np.load(label_path)
    return data, label

# test
# data_path = '../data/AGO1/train'
# data, label = read_file(data_path)
# print data[0], '\t', label[0]
# print data.shape, label.shape

# d,l = read_dirs('../data')
# print d.shape
# cnt = 0
# for i in range(20000):
#     print len(d[i]),'  ',
# print 'cur: ', cnt