import csv
import numpy as np


class ID3:
    def __init__(self):
        pass

    @staticmethod
    def read_data(file):
        raw_data = np.genfromtxt(file, delimiter=',')
        features = np.zeros((raw_data.shape[0], len(raw_data[0])-1))
        labels = np.zeros((raw_data.shape[0]), dtype=np.long)
        for sample in range(raw.shape[0]):
            label[sample] = (1 if my_data[sample][0] == b'M' else 0)
            for feature in range(30):
                new_data[sample, feature + 1] = my_data[sample][feature + 1]

    @staticmethod
    def calc_regression_tree(data)->int:
        max_ig = 0
        res_index = 0

        for i in range (data[0]):

    @staticmethod
    def fit_predict(train, test):

        # Read the train data

        new_data = np.zeros((my_data.shape[0], len(my_data[0])))

        for sample in range(my_data.shape[0]):
            new_data[sample, 0] = (1 if my_data[sample][0] == b'M' else 0)
            for feature in range(30):
                new_data[sample, feature + 1] = my_data[sample][feature + 1]

        print(new_data)
        arr_feature = []
        for i in range(30):
            unsorted_data = new_data[:, [0, i + 1]]
            sorted_data = unsorted_data[unsorted_data[:, 1].argsort()]
            arr_feature.append(sorted_data)
            print(sorted_data)

        a

if '__main__' == __name__:
    ID3.fit_predict("train.csv", None)
