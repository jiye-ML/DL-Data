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
        self.train_images, self.train_labels = self.load(["data_batch_{}".format(i) for i in range(1, 6)])
        self.test_images, self.test_labels = self.load(["test_batch"])
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
        images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1).astype(float) / 255
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



def create_cifar_image():
    cifar10_data = Cifar10Data()
    print("Number of train images: {}".format(len(cifar10_data.train_images)))
    print("Number of train labels: {}".format(len(cifar10_data.train_labels)))
    print("Number of test images: {}".format(len(cifar10_data.test_images)))
    print("Number of test labels: {}".format(len(cifar10_data.test_labels)))
    cifar10_data.display_cifar(10)


if __name__ == "__main__":
    create_cifar_image()
