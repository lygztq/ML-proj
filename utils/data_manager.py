import numpy as np
from load_data import read_file


class DataManager(object):
    """
    The class of data manager
    """
    def __init__(self, dataset_path):
        """
            :param dataset_path: The path of the dataset
        """
        self.path = dataset_path
        self.data, self.label = read_file(self.path)
        self.num_data = self.data.shape[0]


    def next_batch(self, batch_size=128, replace=False):
        """
        Get the next batch of data
            :param batch_size:  The size of a batch of data
            :param replace:     If Ture, no duplicated data
        """
        mask = np.random.choice(range(self.num_data), batch_size, replace=replace)
        return self.data[mask], self.label[mask]


# test
# mng = DataManager('../data/AGO1/train')
# next_b, next_l = mng.next_batch()
# print next_b.shape, next_l.shape
# print next_b[0], '\t', next_l[0]