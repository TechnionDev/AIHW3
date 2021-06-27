import numpy as np
from ID3 import ID3
from ID3 import NUM_FEATURES
from sklearn.model_selection import KFold


def read_train(file):
    """
    Reads the data given in the file for train, returns the data
    :param file: the file name
    :return: the data stored in the file as a numpy array
    """
    raw_data = np.genfromtxt(file, delimiter=',', dtype="S1," + "f8," * (NUM_FEATURES - 1) + "f8")
    new_data = np.zeros((raw_data.shape[0], (NUM_FEATURES + 1)))
    for sample in range(raw_data.shape[0]):
        new_data[sample, 0] = (1 if raw_data[sample][0] == b'M' else 0)
        for feature in range(30):
            new_data[sample, feature+1] = raw_data[sample][feature + 1]
    return new_data


if '__main__' == __name__:

    # Read the data
    data = read_train("train.csv")

    for pruning in range(16):

        # Divide into test and train via KFold
        kf = KFold(n_splits=5, random_state=206560856, shuffle=True)
        test_acc = 0
        train_acc = 0
        for train_index, test_index in kf.split(data):

            train = data[train_index]
            test = data[test_index]

            predictions = ID3.fit_predict(train, test[:, 1:], pruning)

            correct = np.sum(predictions == test[:, 0])
            test_acc += (correct / len(test))

            predictions = ID3.fit_predict(train, train[:, 1:], pruning)
            train_acc += (np.sum(predictions == train[:, 0]) / len(train))

        print(f'pruning {pruning}: error={round(test_acc / 5, 3)} and {round(train_acc / 5, 3)}')
