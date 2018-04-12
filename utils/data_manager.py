import numpy as np
from load_data import read_file


class DataManager(object):
    """
    The class of data manager
    """
    def __init__(self, dataset_path, ratio=0.1):
        """
            :param dataset_path:    The path of the dataset
            :param ratio:           (validation_set_size) / (total_dataset_size)
        """
        self.path = dataset_path
        self.ratio = ratio
        self.data, self.label = read_file(self.path)
        self.num_data = self.data.shape[0]
        self.train_val_split()
        
    def train_val_split(self):
        """
        Split the training set and validation set
        """
        idx = np.arange(self.num_data)
        np.random.shuffle(idx)
        val_num = int(self.ratio * self.num_data)

        self.val_data = self.data[idx[:val_num]]
        self.val_label = self.label[idx[:val_num]]
        self.train_data = self.data[idx[val_num:]]
        self.train_label = self.label[idx[val_num:]]


    def next_train_batch(self, batch_size=128, replace=False):
        """
        Get the next batch of training data
            :param batch_size:  The size of a batch of data
            :param replace:     If Ture, no duplicated data
        """
        mask = np.random.choice(range(self.train_data.shape[0]), batch_size, replace=replace)
        return self.train_data, self.train_label

    def next_val_batch(self, batch_size=32, replace=False):
        """
        Get the next batch of validation data
            :param batch_size:  The size of a batch of data
            :param replace:     If Ture, no duplicated data
        """
        mask = np.random.choice(range(self.val_data.shape[0]), batch_size, replace=replace)
        return self.val_data[mask], self.val_label[mask]        

# test
# mng = DataManager('../data/AGO1/train')
# next_b, next_l = mng.next_train_batch()
# print next_b.shape, next_l.shape
# print next_b[0], '\t', next_l[0]

# print mng.train_data.shape
# print mng.val_data.shape
# print (mng.val_data.shape[0]) * 1.0 / (mng.val_data.shape[0] + mng.train_data.shape[0])