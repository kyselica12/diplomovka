from __future__ import annotations
import numpy as np


class Tracklet:

    def __init__(self, data, data_pred, total_bad, start_idx, end_idx):
        self.data = data
        self.data_pred = data_pred
        self.total_bad = total_bad
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.vector_confidence_average = None
        self.average_vector = None
        self.average_vector_norm = None
        self.gap_confidence = None

        self.compute_confidence()

    def compute_confidence(self):
        sequence_length = self.end_idx - self.start_idx

        vector = self.data[2:] - self.data[1:-1]
        vector_pred = self.data_pred[2:] - self.data_pred[1:-1]

        vector_norm = np.linalg.norm((vector_pred), axis=1)
        vector_confidence = np.abs(vector_norm - np.linalg.norm((vector - vector_pred), axis=1)) / vector_norm
        dif = np.abs(self.data - self.data_pred)
        ok = np.logical_and(dif[:, 0] != 0, dif[:, 1] != 0)[2:]

        self.vector_confidence_average = np.mean(vector_confidence[ok])
        self.average_vector = np.mean(vector, axis=0)
        self.average_vector_norm = np.linalg.norm(self.average_vector)
        self.gap_confidence = (sequence_length - 2 - self.total_bad) / (sequence_length - 2)

    def append(self, t: Tracklet) -> Tracklet:
        start_idx = self.start_idx
        end_idx = t.end_idx

        data = np.concatenate((self.data, t.data[self.end_idx-t.start_idx:]))
        data_pred = np.concatenate((self.data_pred, t.data_pred[self.end_idx-t.start_idx:]))
        total_bad = self.total_bad + t.total_bad

        return Tracklet(data, data_pred, total_bad, start_idx, end_idx)


    def get_last_n(self, n):
        data = self.data[-n:]
        return data

    def get_last_n_bad_in_row(self, n):
        bad = self.data[-n:] == self.data_pred[-n:]
        res = 0
        for x in bad[::-1]:
            if np.all(x):
                res += 1
            else:
                break
        return res

    def is_the_same(self, other: Tracklet):
        overlap = other.start_idx - self.start_idx
        ok = self.data[overlap:] != self.data_pred[overlap:] # data from images not guessed
        return np.all(self.data[overlap:][ok] == other.data[:self.end_idx-overlap][ok])

    # TODO -> spravit finalny vypis do suboru, bude treba pracovat s celou sekvenciou

