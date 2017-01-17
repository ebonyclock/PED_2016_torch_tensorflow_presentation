import numpy as np
from matplotlib import pyplot as plt

plt.style.use('ggplot')


def generate_circular_points(u, o, n):
    mean = u
    cov = [[o[0], 0], [0, o[1]]]

    return np.random.multivariate_normal(mean, cov, n)


def get_dataset_1(n=300):
    ds1 = generate_circular_points([0, 0], [1, 1], n)
    ds2 = generate_circular_points([2.5, 2.5], [1, 1], n)
    dataset = np.concatenate([ds1, ds2], axis=0)
    labels = np.zeros((2*n), dtype=np.int8)
    labels[n:] = 1
    return mix((dataset, labels))


def get_dataset_2(n=300):
    n *= 2
    ds1 = generate_circular_points([0, 0], [1, 1], n)
    d2 = ds1 ** 2
    d2 = d2.sum(axis=1)
    labels = np.zeros((n), dtype=np.int8)
    labels[d2 > 1.2] = 1
    return ds1, labels


def plot_data(data_set):
    cor, lab = data_set
    for i in range(max(lab)+1):
        x, y = cor[lab[:, 0] == i].T
        plt.plot(x, y, 'x')

    plt.axis('equal')
    plt.show()


def plot_predicted_data(data_set, predicted_labels):
    true_positive = []
    false_positive = []
    true_negative = []
    false_negative = []

    for i, [[x, y], [l]] in enumerate(zip(*data_set)):
        if l == 1.0 and predicted_labels[i]:
            true_positive.append([x, y])
        elif l == 1.0:
            false_negative.append([x, y])
        elif l == 0.0 and predicted_labels[i]:
            false_positive.append([x, y])
        else:
            true_negative.append([x, y])

    if true_positive:
        plt.plot(np.array(true_positive).T[0], np.array(true_positive).T[1], 'o', color='#0c00ff')
    if true_negative:
        plt.plot(np.array(true_negative).T[0], np.array(true_negative).T[1], 'o', color='#5cce00')
    if false_negative:
        plt.plot(np.array(false_negative).T[0], np.array(false_negative).T[1], 'o', color='#ff002a')
    if false_positive:
        plt.plot(np.array(false_positive).T[0], np.array(false_positive).T[1], 'o', color='#aa0047')

    plt.axis('equal')
    plt.show()


def plot_dataset(dataset, val=False):
    if val:
        plot_data(dataset.valid)
    else:
        plot_data(dataset.train)


def mix(ds):
    ds_copy = np.concatenate([ds[0], ds[1].reshape((-1, 1))], axis=1)
    np.random.shuffle(ds_copy)
    return ds_copy[:, 0:2], ds_copy[:, 2].astype(np.int8)


def read_file(name, train_proc=0.6):
    f = open(name)

    result = []

    for l in f:
        result.append([float(field) for field in l.split()])

    train_size = int(len(result) * train_proc)

    x = np.array(result, dtype=np.float32)
    return Dataset([x[0:train_size, 0:2], x[0:train_size, 2].reshape(-1, 1)], [x[train_size:, 0:2], x[train_size:, 2].reshape(-1, 1)])


class Dataset:
    def __init__(self, train=None, valid=None):
        self.train = train
        self.valid = valid
        if train is not None:
            self.attributes_number = train[0].shape[1]
        elif valid is not None:
            self.attributes_number = valid[0].shape[1]

    def get_train_batches(self, batchsize=60):
        permutation = np.random.permutation(self.train[0].shape[0])

        for start_i in range(0, len(permutation), batchsize):
            end_i = min(start_i + batchsize, len(self.train[0]))
            rng = permutation[start_i:end_i]
            yield [self.train[0][rng], self.train[1][rng]]

    def get_valid_batches(self, batchsize=60):
        for start_i in xrange(0, self.valid[0].shape[0], batchsize):
            end_i = min(start_i + batchsize, self.valid[0].shape[0])
            yield [self.valid[0][start_i:end_i], self.valid[1][start_i:end_i]]
