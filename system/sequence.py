import glob
import pandas as pd
import numpy as np


class Sequence:

    def __init__(self, path, sufix='_m_s_a'):
        self.seq = self.load(path, sufix)
        self.MIN, self.MAX, self.VAR = self.compute_stats()


    def load(self, path, sufix):
        N = max([int(f[:-4 - len(sufix)][-4:]) for f in glob.iglob(f'{path}/*AR/*.tsv')])

        seq = [np.zeros((0, 2)) for _ in range(N)]

        for tsv in glob.iglob(f'{path}/*AR/*.tsv'):
            num = int(tsv[:-4 - len(sufix)][-4:]) - 1
            res = pd.read_csv(tsv, sep="\t")
            res.columns.name = None
            seq[num] = res.to_numpy()[:, [-2, -1]]

        return seq

    def compute_stats(self):

        s = self.seq[0][np.newaxis]
        for x in self.seq[1:]:
            s = np.append(s, x[np.newaxis])

        s = s.reshape(-1,2)

        MIN = np.min(s, axis=0)
        MAX = np.max(s, axis=0)
        VAR = MAX-MIN

        return MIN, MAX, VAR

    def get_min(self):
        return self.MIN

    def get_var(self):
        return self.VAR

    def get_data(self):
        return self.seq