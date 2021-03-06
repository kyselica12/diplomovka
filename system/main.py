import glob
from typing import List

import torch

from argparser import parse_args
from model import Model
from sequence import Sequence, TestSequence
from systemLoop import SystemLoop

import numpy as np

from tracklet import Tracklet

class System:

    def find_tracklet(self,seq, model, k, start_idx, end_idx, match_threshold, tracklets: List[Tracklet], penalty) -> List[Tracklet]:

        systemLoop = SystemLoop(seq=seq, k=k, start_idx=start_idx, end_idx=end_idx)

        status = systemLoop.create_status(tracklets, penalty=penalty)

        if tracklets is not None:
            init_length = tracklets[0].end_idx - start_idx
        else:
            init_length = 2

        for i, img in systemLoop.loop(status, init_length=init_length):
            model.predict(status)
            systemLoop.match(status, img, threshold=match_threshold)
            systemLoop.update(status)
        res = systemLoop.get_result(status)

        return res

    def combine_tracklets(self,tracklets, tmp):
        if tracklets is None:
            tracklets = tmp
        else:
            res = []
            for t1 in tracklets:
                for idx, t2 in enumerate(tmp):
                    if t1.is_part_of(t2):
                        res.append(t1)
                        tmp = np.delete(tmp, idx)
                        break
                else:
                    res.append(t1)
            tracklets = np.concatenate((tmp, np.array(res)))

        return tracklets

    def process_sequence(self,model, k, overlap,match_threshold, seq):

        tracklets = None
        for i in range(k + 1):
            tmp = self.find_tracklet(seq=seq,
                                model=model,
                                k=k,
                                start_idx=i,
                                end_idx=8,
                                match_threshold=match_threshold,
                                tracklets=None,
                                penalty=i)

            tracklets = self.combine_tracklets(tracklets, tmp)

        print(f'Number of initial tracklets: {len(tracklets)}')

        step = 8 - overlap
        for i in range(step, len(seq)-overlap, step):
            if len(tracklets) == 0:
                print('No object found')
                break
            tracklets = self.find_tracklet(seq=seq,
                                      model=model,
                                      k=k,
                                      start_idx=i,
                                      end_idx=min(i + 8, len(seq)),
                                      match_threshold=match_threshold,
                                      tracklets=tracklets,
                                      penalty=0)
            if len(tracklets) == 0:
                print('No object found')
                break

        return tracklets

    def test(self):
        model_path = "Albert_vector.model"
        masking_threshold = 0.002
        match_threshold = 0.003
        K = 2
        overlap = 2

        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda:0'

        print(f'Load model {model_path}')
        model = Model(path=model_path, device=device)

        for seq_path in glob.iglob(f"./astronomia/*"):

            print(f'Load sequence {seq_path}')
            seq = TestSequence(path=seq_path, masking_threshold=masking_threshold)

            tracklets = self.process_sequence(model, K, overlap, match_threshold, seq)

            found = False
            for r in tracklets:
                print(r.gap_confidence, r.vector_confidence_average)
                print(r.start_idx, r.end_idx, r.average_vector)
                ok = seq.compare_TB_from_file(r)
                if ok:
                    found = True
                    seq.save_tracklets([r])


            print(f'   -> {":)" if found else ":/ Not"} Found')

    def run(self):
        args = parse_args()

        seq_path = "03005A_2_R_27-11-20_10..30..35"  # args.input
        model_path = args.model

        masking_threshold = args.masking
        match_threshold = args.matching
        K = args.k
        overlap = args.overlap

        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda:0'

        print(f'Load model {model_path}')
        model = Model(path=model_path, device=device)

        print(f'Load sequence {seq_path}')
        seq = Sequence(path=seq_path, masking_threshold=masking_threshold)

        tracklets = self.process_sequence(model, K, overlap, match_threshold, seq)

        print(f'Found {len(tracklets)} tracklets')

        seq.save_tracklets(tracklets)

if __name__ == "__main__":

    TEST = False
    system = System()

    if TEST:
        print("Test")
        system.test()
    else:
        system.run()










