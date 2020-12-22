import itertools
from dataclasses import dataclass
import numpy as np
from scipy.spatial import distance

from sequence import Sequence


@dataclass
class LoopStatus:
    data : np.ndarray
    data_pred: np.ndarray
    bad_predictions_in_row: np.ndarray
    total_bad_predictions: np.ndarray
    directions: np.ndarray = None
    predictions: np.ndarray = None
    matched: np.ndarray = None

    def is_empty(self):
        return self.data.shape[0] == 0


class SystemLoop:

    def __init__(self, seq: Sequence, start_idx, end_idx, k=2):

        self.low_bound = seq.get_min()
        self.var = seq.get_var()
        self.k = k

        self.seq = [(x-self.low_bound)/self.var for x in seq.get_data()[start_idx:end_idx]]

    def loop(self, status: LoopStatus):
        for img in self.seq[2:]:
            if status.is_empty():
                break
            yield img

    def create_status(self):

        data = self.create_couples()
        data_pred = data.copy()
        bad_predictions_in_row = np.zeros((data.shape[0]), dtype=int)
        total_bad_predictions = np.zeros((data.shape[0]), dtype=int)

        status = LoopStatus(data, data_pred, bad_predictions_in_row, total_bad_predictions)

        return status

    def create_couples(self):
        couples = [ list(item) for item in itertools.product(self.seq[0], self.seq[1])]
        return np.array(couples)

    def masking(self, threshold):
        n = self.seq[0].shape[1] + 1
        data = []

        for i, image in enumerate(self.seq):
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
                X_dist = data[i][0] - data[obj][0]   # distance in X axis  [%]
                Y_dist = data[i][1] - data[obj][1]   # distance in Y axis  [%]
                dist = np.sqrt(X_dist ** 2 + Y_dist ** 2)

                if dist > threshold:
                    obj = i
                    indexes.append(obj)

        data = data[indexes]

        new_seq = [None] * len(self.seq)

        for i in range(len(self.seq)):
            new_seq[i] = data[data[:, n - 1] == i][:, :-1]

        self.seq = new_seq

    def match(self, status: LoopStatus, img, threshold):

        img = img[np.argsort(img[:, 0])]   # translate position from [px] to [%]
        predictions = status.predictions

        matched = - np.ones(predictions.shape)


        for i, p in enumerate(predictions):
            x, y = p
            candidates = img[(x - threshold < img[:, 0]) * (img[:, 0] < x + threshold)]

            if len(candidates) == 0:
                continue

            dist = distance.cdist([[x, y]], candidates)[0]

            min_idx = np.argmin(dist)

            if dist[min_idx] < threshold:
                matched[i] = candidates[min_idx]

        status.matched = matched

    def update(self, status: LoopStatus):

        found = status.matched[:, 0] > 0
        not_found = np.logical_not(found)

        status.total_bad_predictions += not_found
        status.bad_predictions_in_row = (status.bad_predictions_in_row + not_found) * not_found

        ok = np.logical_or(found, status.bad_predictions_in_row <= self.k)

        status.matched[not_found] = status.predictions[not_found]

        status.data = np.concatenate((status.data[ok], status.matched[ok].reshape(-1, 1, 2)), axis=1)
        status.data_pred = np.concatenate((status.data_pred[ok], status.predictions[ok].reshape(-1, 1, 2)), axis=1)

        status.total_bad_predictions = status.total_bad_predictions[ok]
        status.bad_predictions_in_row = status.bad_predictions_in_row[ok]

    def get_result(self, status: LoopStatus):

        if status.is_empty():
            return []


        data = status.data*self.var + self.low_bound
        data_pred = status.data_pred * self.var + self.low_bound
        sequence_length = data[0].shape[0]

        vector = data[:,2:] - data[:,1:-1]
        vector_pred = data_pred[:,2:] - data_pred[:,1:-1]

        vector_norm = np.linalg.norm((vector_pred), axis=2)
        vector_confidence = np.abs(vector_norm - np.linalg.norm((vector - vector_pred), axis=2)) / vector_norm
        dif = np.abs(data-data_pred)
        ok = np.logical_and(dif[:,2:, 0] != 0, dif[:,2:, 1] != 0)

        vector_confidence_average = [np.mean(vector_confidence[i][ok[i]]) for i in range(len(dif))]
        average_vector = np.mean(vector, axis=1)
        average_vector_norm = np.linalg.norm(average_vector, axis=1)
        gap_confidence = (sequence_length - 2 - status.total_bad_predictions) / (sequence_length-2)

        res = []
        for i in range(len(data)):
            res.append((data[i], data_pred[i], (average_vector[i], average_vector_norm[i]),(vector_confidence_average[i], gap_confidence[i])))

        return res
