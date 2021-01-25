import glob
from os import path
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tracklet import Tracklet


class Sequence:

    def __init__(self, path, sufix='_m_s_a', masking_threshold=0.002):
        self.names = None
        self.path = path
        self.sufix = sufix
        self.seq = self.load()
        self.MIN, self.MAX, self.VAR = self.compute_stats()
        self.masked_normalized_data = self.masking(masking_threshold)

    def load(self):
        N = max([int(f[:-4 - len(self.sufix)][-4:]) for f in glob.iglob(f'{self.path}/*AR/*.tsv')])

        seq = [np.zeros((0, 2)) for _ in range(N)]

        for tsv in glob.iglob(f'{self.path}/*AR/*.tsv'):
            num = int(tsv[:-4 - len(self.sufix)][-4:]) - 1
            res = pd.read_csv(tsv, sep="\t")
            if self.names is None:
                self.names = list(res.columns)
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

    def masking(self, threshold=0.002):
        seq = [(x[:, -2:] - self.MIN) / self.VAR for x in self.seq ]
        n = seq[0].shape[1] + 1
        data = []

        for i, image in enumerate(seq):
            image = np.concatenate((image, np.ones((image.shape[0], 1)) * i), axis=1)
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

        new_seq = [None] * len(seq)

        for i in range(len(seq)):
            new_seq[i] = data[data[:, n - 1] == i][:, :-1]

        return new_seq

    def get_normalized_equatorial_data(self, start, end):
        return self.masked_normalized_data[start:end].copy()

    def interprate_tracklet(self, t: Tracklet):

        data = t.data

        seq_idx = []

        for i in range(t.start_idx, t.end_idx):
            ok = self.seq[i][:, -2:] == data[i-t.start_idx]
            point_position = np.logical_and(ok[:, 0], ok[:, 1])
            if np.all(np.logical_not(point_position)):
                continue
            seq_idx.append([i, np.where(point_position)[0][0]])

        return seq_idx

    def get_data_from_index(self, idx):
        return np.array([self.seq[x[0]][x[1]] for x in idx])

    def save_tracklets(self, tracklets: List[Tracklet]):

        for i, t in enumerate(tracklets):
            idx = np.array(self.interprate_tracklet(t))
            data = np.array([self.seq[i][j] for i, j in idx])

            data = np.concatenate((idx[:, 0].reshape(-1,1),data), axis=1)

            names = ["file"] + self.names
            df = pd.DataFrame(data=data, index=data[:,0], columns=names)

            seq_name = self.path[-30:-20]
            df['file'] = df['file'].apply(lambda x: f'{seq_name}-{int(x) + 1:04}{self.sufix}')

            df.to_csv(f'{self.path}/{seq_name}_TB_{i:02}.tsv', sep='\t', index=False)

            df = pd.DataFrame(data=np.array([[t.gap_confidence, t.vector_confidence_average]]), columns=['match_confidence', 'vector_confidence'])
            df.to_csv(f'{self.path}/{seq_name}_TB_{i:02}_confidence.tsv', sep='\t', index=False)


    def __len__(self):
        return len(self.seq)

class TestSequence(Sequence):

    def load_TB_file(self):
        name = self.path[-30:-20]
        txt = f'{self.path}/IPE-TB/{name}.txt'

        if not path.exists(txt):
            raise FileNotFoundError("Tracklet building file does not exist")

        with open(txt, 'r') as f:
            lines = f.read().split('\n')

        idx = []

        for line in lines[18:]:
            if line == '':
                break

            parts = line.split()
            num = int(parts[0][11:15])
            x = float(parts[9])
            y = float(parts[10])

            ok = np.round(self.seq[num-1][:, :2], decimals=2) == [x, y]
            t = self.seq[num-1][ok[:, 0] * ok[:, 1]][0]  # prvy_stpec_zhodny AND druhy_stlpec_zhodny

            idx.append([num-1, np.where(ok[:, 0] * ok[:, 1])[0][0]])

        return np.array(idx)

    def compare_TB_from_file(self, t: Tracklet):

        if not hasattr(self, 'TB'):
            self.TB = self.load_TB_file()

        idx = self.interprate_tracklet(t)

        for x in self.TB:
            if not np.any(x == idx):
                return False

        print(f'Diference in length: {len(idx)-len(self.TB)}')

        t1 = set([tuple(x) for x in idx])
        t2 = set([tuple(x) for x in self.TB])

        in_t1_only = self.get_data_from_index(t1 - t2)
        in_t2_only = self.get_data_from_index(t2 - t1)
        in_t1_t2 = self.get_data_from_index(t1 & t2)

        fig, ax = plt.subplots()
        if len(in_t1_only) > 0: self.plot_points(ax, in_t1_only, 'Only in found tracklet', 'blue')
        if len(in_t2_only) > 0: self.plot_points(ax, in_t2_only, 'Only in orginal tracklet', 'red')
        if len(in_t1_t2)   > 0: self.plot_points(ax, in_t1_t2, 'In both tracklets', 'green')
        ax.legend()

        fig.savefig(f'{self.path}/{self.path[-30:-20]}_TB.png')
        plt.close(fig)

        return True

    def plot_points(self, ax, data, label, color):
        dataX, dataY = [],[]
        if len(data) > 0:
            dataX = data[:, -1]
            dataY = data[:, -2]

        ax.scatter(dataX, dataY, label=label, color=color)