import torch
from model import Model
from sequence import Sequence
from systemLoop import SystemLoop

import numpy as np

if __name__ == "__main__":
    seq_path = "03005A_2_R_27-11-20_10..30..35"
    model_path = "Albert_vector.model"

    device = 'cpu'
    if torch.cuda.is_available():
        device='cuda:0'

    print('Load model')
    model = Model(path=model_path, device=device)

    print('Load sequence')
    seq = Sequence(path=seq_path)

    print('Create system loop')
    systemLoop = SystemLoop(seq=seq, start_idx=0, end_idx=8, k=2)

    print('Masking')
    systemLoop.masking(threshold=0.002)
    status = systemLoop.create_status()

    for img in systemLoop.loop(status):
        print('Loop')
        model.predict(status)
        systemLoop.match(status, img, threshold=0.005)
        systemLoop.update(status)

        # r = status.data[np.abs((status.data*systemLoop.var + systemLoop.low_bound)[:,0,0] - 49.24844216) < 0.001]
        # print(r*systemLoop.var + systemLoop.low_bound)


    res = systemLoop.get_result(status)

    print(res)