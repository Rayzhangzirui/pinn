#!/usr/bin/env python
import sys
sys.path.insert(1, '/home/ziruz16/pinn')
from scipy.io import loadmat
import numpy as np

if __name__ == "__main__":
    matdat = loadmat(sys.argv[1])
    for k,v in matdat.items():
        if isinstance(v, np.ndarray):
            print(f'{k} {v.shape}')

    
