import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

STEPS = 500000

class Cifar10Data:

    def __init__(self, data_path = "cifar-10-batches-py", batch_size = 50):
        self._data_path = data_path
        self._batch_size = batch_size

        self._i = 0

        # data
        self.train_images, self.train_labels = self._load(["data_batch_{}".format(i) for i in range(1, 6)])
        self.test_images, self.test_labels = self._load(["test_batch"])
        pass

    @staticmethod
    def one_hot(vec, vals=10):
        n = len(vec)
        out = np.zeros((n, vals))
        out[range(n), vec] = 1
        return out

    def _unpickle(self, file):
        with open(os.path.join(self._data_path, file), 'rb') as fo:
            return pickle.load(fo, encoding='latin1')

    def _load(self, source):
        data = [self._unpickle(f) for f in source]
        images = np.vstack([d["data"] for d in data])
        n = len(images)
        images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255
        labels = Cifar10Data.one_hot(np.hstack([d["labels"] for d in data]), 10)
        return images, labels

    def next_batch(self, batch_size):
        x, y = self.test_images[self._i : self._i + batch_size], self.test_labels[self._i : self._i+batch_size]
        self._i = (self._i + batch_size) % len(self.test_images)
        return x, y

    # 训练阶段，随机选取batch_size个
    def random_batch(self):
        ix = np.random.choice(len(self.train_images), self._batch_size)
        return self.train_images[ix], self.train_labels[ix]

    def display_cifar(self, size):
        n = len(self.train_images)
        plt.figure()
        plt.gca().set_axis_off()
        im = np.vstack([np.hstack([self.train_images[np.random.choice(n)] for i in range(size)]) for i in range(size)])
        plt.imshow(im)
        plt.show()

    pass


class PreData:

    def __init__(self, data, flip=True, shuffle=True, _standardization=True):
        self.data = data
        self.flip = flip
        self.shuffle = shuffle
        self.standardization = _standardization
        pass

    def __call__(self, *args, **kwargs):
        train_data, train_labels = self.data.train_images, self.data.train_labels
        test_data, test_labels = self.data.test_images, self.data.test_labels

        print("Train data:", np.shape(train_data), np.shape(train_labels))
        print("Test data :", np.shape(test_data), np.shape(test_labels))
        print("======Load finished======")

        if self.shuffle:
            print("======Shuffling data======")
            indices = np.random.permutation(len(train_data))
            train_data = train_data[indices]
            train_labels = train_labels[indices]
        if self.standardization:
            print("======color_preprocess data======")
            train_data, test_data = self._standardization(train_data, test_data)
        if self.flip :
            train_data += self._random_flip_leftright(train_data)

        return train_data, train_labels, test_data, test_labels

    def _random_flip_leftright(self, batch):
        for i in range(len(batch)):
            if bool(np.random.getrandbits(1)):
                batch[i] = np.fliplr(batch[i])
        return batch

    def _standardization(self, x_train, x_test):
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std = [62.9932, 62.0887, 66.7048]
        for i in range(3):
            x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
            x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
        return x_train, x_test

    pass


def create_cifar_image():
    cifar10_data = Cifar10Data()
    print("Number of train images: {}".format(len(cifar10_data.train_images)))
    print("Number of train labels: {}".format(len(cifar10_data.train_labels)))
    print("Number of test images: {}".format(len(cifar10_data.test_images)))
    print("Number of test labels: {}".format(len(cifar10_data.test_labels)))
    cifar10_data.display_cifar(10)


if __name__ == "__main__":
    create_cifar_image()
