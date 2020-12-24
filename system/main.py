from typing import List

import torch
from model import Model
from sequence import Sequence
from systemLoop import SystemLoop

import numpy as np

from tracklet import Tracklet


def find_tracklet(seq, model, k, start_idx, end_idx, tracklets: List[Tracklet], penalty) -> List[Tracklet]:

    print('Create system loop')
    systemLoop = SystemLoop(seq=seq, k=K, start_idx=start_idx, end_idx=end_idx)
    print('Masking')
    systemLoop.masking(threshold=0.002)
    status = systemLoop.create_status(tracklets, penalty=penalty)

    if tracklets is not None:
        init_length = tracklets[0].end_idx - start_idx
    else:
        init_length = 2

    for i, img in systemLoop.loop(status, init_length=init_length):
        model.predict(status)
        systemLoop.match(status, img, threshold=0.01)
        systemLoop.update(status)
    res = systemLoop.get_result(status)

    return res


def combine_tracklets(tracklets, tmp):
    if tracklets is None:
        tracklets = tmp
    else:
        res = []
        for t1 in tracklets:
            for idx, t2 in enumerate(tmp):
                if t1.is_the_same(t2):
                    res.append(t1)
                    tmp = np.delete(tmp, idx)
                    break
            else:
                res.append(t1)
        tracklets = np.concatenate((tmp, np.array(res)))

    return tracklets


# TODO vybrat parametre von
# - masking threshold
# - match threshold
# - K
# - overlap

if __name__ == "__main__":
    seq_path = "10007C_5_R_30-11-20_10..21..28"
    model_path = "Albert_vector.model"

    device = 'cpu'
    if torch.cuda.is_available():
        device='cuda:0'

    K = 2

    print('Load model')
    model = Model(path=model_path, device=device)

    print('Load sequence')
    seq = Sequence(path=seq_path)

    tracklets = None
    for i in range(K + 1):
        tmp = find_tracklet(seq, model, K, start_idx=i, end_idx=8, tracklets=None, penalty=i)
        tracklets = combine_tracklets(tracklets, tmp)


    print(f'Number of tracklets: {len(tracklets)}')

    for i in range(6, len(seq)-6, 6):
        tracklets = find_tracklet(seq, model, K, start_idx=i, end_idx=i+8, tracklets=tracklets, penalty=0)

        if len(tracklets) == 0:
            print('No object found')
            break

    for r in tracklets:
        print(r.data, r.gap_confidence, r.vector_confidence_average)









