import numpy as np
import random
from random import randrange as rr
from tqdm import tqdm


class DataGenerator(object):
    mean_x, mean_y = 512, 512
    std_x, std_y = 512, 512

    def __init__(self, batch_size, n_batches, k, n_points=0, validation=False):
        self.length = n_batches if not validation else max(1, n_batches // 5)
        self.batch_size = batch_size
        self.k = k
        self.n_points = n_points

        self.data = [self.generate_batch() for _ in tqdm(range(self.length))]

    def generate_batch(self):
        X, y = [], []
        for _ in range(self.batch_size):
            e = self.generate_example()
            X.append(e[0])
            y.append(e[1])

        X, y = np.array(X).transpose(1, 0, 2), np.array(y)[np.newaxis]

        return self.normalize(X, y)

    def generate_example(self):
        def sgn(x):
            return -1 if x < 0 else 1

        x = rr(24, 1000)
        y = rr(24, 1000)

        dir_x = 0 if random.random() < 0.5 else 1024
        dir_y = 0 if random.random() < 0.5 else 1024

        delta_x = rr(1, abs(dir_x - x) // 8) * sgn(dir_x - x)
        delta_y = rr(1, abs(dir_y - x) // 8) * sgn(dir_y - x)

        point = np.array([x, y])
        delta = np.array([delta_x, delta_y])

        data = point + np.outer(delta,
                                np.linspace(start=0, stop=self.k - 1, endpoint=True, num=self.k + self.n_points)).T

        return data, data[-1] + [delta_x, delta_y]

    def normalize(self, X, y):
        return (X - self.mean_x) / self.std_x, (y - self.mean_y) / self.std_y

    def denormalize(self, X, y):
        return X * self.std_x + self.mean_x, y * self.std_y + self.mean_y

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.length

    @staticmethod
    def normalizeInput(X):
        return (X - DataGenerator.mean_x) / DataGenerator.std_x

    @staticmethod
    def denormalizeOutput(y):
        return y * DataGenerator.std_y + DataGenerator.mean_y


class DataGeneratorVector(DataGenerator):
    mean_x, mean_y = 512, 0
    std_x, std_y = 512, 1028 / 8

    def __init__(self, batch_size, n_batches, k, n_points=0, validation=False, multi_k=False):
        self.length = n_batches if not validation else max(1, n_batches // 5)
        self.batch_size = batch_size
        self.k = k
        self.n_points = n_points
        self.multi_k = multi_k

        self.data = [self.generate_batch() for _ in tqdm(range(self.length))]

    def generate_batch(self):

        X, y = [], []

        if self.multi_k:
            self.k = rr(2, 8)

        for _ in range(self.batch_size):
            e = self.generate_example()
            X.append(e[0])
            y.append(e[1])

        X, y = np.array(X).transpose(1, 0, 2), np.array(y)[np.newaxis]

        return self.normalize(X, y)

    def generate_example(self):

        def sgn(x):
            return -1 if x < 0 else 1

        x = rr(24, 1000)
        y = rr(24, 1000)

        dir_x = 0 if random.random() < 0.5 else 1024
        dir_y = 0 if random.random() < 0.5 else 1024

        delta_x = rr(1, abs(dir_x - x) // 8) * sgn(dir_x - x)
        delta_y = rr(1, abs(dir_y - x) // 8) * sgn(dir_y - x)

        point = np.array([x, y])
        delta = np.array([delta_x, delta_y])

        data = point + np.outer(delta,
                                np.linspace(start=0, stop=self.k - 1, endpoint=True, num=self.k + self.n_points)).T

        return data, [delta_x, delta_y]

    @staticmethod
    def normalizeInput(X):
        return (X - DataGeneratorVector.mean_x) / DataGeneratorVector.std_x

    @staticmethod
    def denormalizeOutput(y):
        return y * DataGeneratorVector.std_y + DataGeneratorVector.mean_y