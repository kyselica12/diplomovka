import glob
import pandas as pd
import numpy as np


class Sequence:

    def __init__(self, path, sufix='_m_s_a'):
        self.names = None
        self.seq = self.load(path, sufix)
        self.masked = self.masking(0.002)
        self.MIN, self.MAX, self.VAR = self.compute_stats()



    def load(self, path, sufix):
        N = max([int(f[:-4 - len(sufix)][-4:]) for f in glob.iglob(f'{path}/*AR/*.tsv')])

        seq = [np.zeros((0, 2)) for _ in range(N)]

        for tsv in glob.iglob(f'{path}/*AR/*.tsv'):
            num = int(tsv[:-4 - len(sufix)][-4:]) - 1
            res = pd.read_csv(tsv, sep="\t")
            if self.names is None:
                self.names = res.columns.name
            res.columns.name = None
            seq[num] = res.to_numpy()

        return seq

    def compute_stats(self):

        s = self.seq[0][:, -2:][np.newaxis]
        for x in self.seq[1:]:
            s = np.append(s, x[:,-2:][np.newaxis])

        s = s.reshape(-1,2)

        MIN = np.min(s, axis=0)
        MAX = np.max(s, axis=0)
        VAR = MAX-MIN

        return MIN, MAX, VAR

    def get_min(self):
        return self.MIN

    def get_var(self):
        return self.VAR

    def get_equitorial_data(self):
        return [ x[:, -2:] for x in self.seq ]

    def masking(self, threshold):
        # TODO opravit a vymysliet ako to vratit pre SystemLoop
        n = self.seq[0].shape[1] + 1
        data = []

        for i, image in enumerate(self.seq):
            image = np.concatenate(((image[:,-2:]-self.MIN)/self.VAR, np.ones((image.shape[0], 1)) * i), axis=1)
            data = np.append(data, image)

        data = data.reshape(-1, n)

        # sort along X-axis

        indexes = []
        obj = None
        dup_flag = False

        for i in np.lexsort((data[:, 1], data[:, 0])):
            # take first object as reference
            if obj is None:
                obj = i
                indexes.append(obj)
            else:
                X_dist = data[i][0] - data[obj][0]  # distance in X axis  [%]
                Y_dist = data[i][1] - data[obj][1]  # distance in Y axis  [%]
                dist = np.sqrt(X_dist ** 2 + Y_dist ** 2)

                if dist > threshold:
                    obj = i
                    indexes.append(obj)

        data = data[indexes]

        new_seq = [None] * len(self.seq)

        for i in range(len(self.seq)):
            new_seq[i] = self.seq[data[:, n - 1] == i]

        return new_seq

    def __len__(self):
        return len(self.seq)