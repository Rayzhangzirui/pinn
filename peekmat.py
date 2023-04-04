#!/usr/bin/env python
import sys
sys.path.insert(1, '/home/ziruz16/pinn')
from scipy.io import loadmat
import numpy as np

import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script with two optional command line arguments.")
    parser.add_argument('file', type=str, help="A positional argument.")
    parser.add_argument('-p', '--printval', action='store_true', help="print value")
    parser.add_argument('-v', '--varname', type=str, default="", help="which variable")

    args = parser.parse_args()
    matdat = loadmat(args.file,mat_dtype=True)

    if args.varname:
        print(matdat[args.varname].shape)
        if args.printval:
            print(matdat[args.varname])
    else:
        for k,v in matdat.items():
            if isinstance(v, np.ndarray):
                print(f'{k} {v.shape} {v.dtype}')
                if args.printval:
                    print(f'{v}')

    
