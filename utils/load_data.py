import numpy as np

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

# test
# data_path = '../data/AGO1/train'
# data, label = read_file(data_path)
# print data[0], '\t', label[0]
# print data.shape, label.shape