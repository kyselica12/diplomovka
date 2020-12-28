import argparse

def parse_args():


    parser = argparse.ArgumentParser(description='Tracklet building')

    parser.add_argument('-M','--model',
                        action='store',
                        default=str(),
                        type=str,
                        required=True,
                        help='file with saved model')

    parser.add_argument('-I','--input',
                        action='store',
                        default=str(),
                        type=str,
                        required=True,
                        help='Sequence folder')

    parser.add_argument('-k',
                        action='store',
                        default=1,
                        type=int,
                        required=False,
                        help='Maximum number of images in row without object in tracklet')

    parser.add_argument('-S','--masking',
                        action='store',
                        default=0.002,
                        type=float,
                        required=False,
                        help='Masking threshold from 0 to 1 -> 0.01 means thresholf is 1% of size of image')

    parser.add_argument('-T','--matching',
                        action='store',
                        default=0.01,
                        type=float,
                        required=False,
                        help='Matching threshold from 0 to 1 -> 0.01 means thresholf is 1% of size of image')

    parser.add_argument('-O','--overlap',
                        action='store',
                        default=2,
                        type=int,
                        required=False,
                        help='Overlap size in tracklet building')

    return parser.parse_args()


if __name__ == "__main__":

    file = "03005A_2_R_27-11-20_10..30..35"

    args = parse_args()

    print(args.input)