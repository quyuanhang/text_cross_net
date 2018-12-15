import tqdm
import argparse
import pandas as pd
import numpy as np


def binary(s):
    if int(s) > 0:
        return 1
    return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datain')
    parser.add_argument('--dataout')
    parser.add_argument('--rand_sample', type=int, default=1)
    args = parser.parse_args()

    train_frame = pd.read_csv(
        '{}.train'.format(args.datain),
        sep='\001',
        header=None,
        dtype=str,
    )
    train_frame.columns = ['eid', 'jid', 'label']
    train_frame['label'] = train_frame['label'].map(binary)

    if args.rand_sample:
        train_frame, test_frame = train_frame.iloc[:-20000], train_frame.iloc[-20000:]
    else:
        test_frame = pd.read_csv(
            '{}.test'.format(args.datain),
            sep='\001',
            header=None,
            dtype=str,
        )
        test_frame = test_frame.iloc[:20000]
        test_frame.columns = ['eid', 'jid', 'label']
        test_frame['label'] = test_frame['label'].map(binary)

    train_frame.to_csv('{}.train'.format(args.dataout), sep='\001', header=None, index=False)
    test_frame.to_csv('{}.test'.format(args.dataout), sep='\001', header=None, index=False)



