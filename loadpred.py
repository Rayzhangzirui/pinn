#!/usr/bin/env python
# load model and make prediction

import sys
sys.path.insert(1, '/home/ziruz16/pinn')

from config import *
from glioma import *
import json
import argparse


if __name__ == "__main__":
    
    optfile = sys.argv[1] #option file, used to load model
    
    parser = argparse.ArgumentParser(description="load option, make prediction")
    parser.add_argument("optfile", type=str, help="option file, used to load model")
    parser.add_argument("-p", "--path", type=str, help="Path to the ckpt file")
    parser.add_argument("-n", "--name", type=str, help="Name of the output prediciton file")

    args = parser.parse_args()
    # Opening JSON file
    with open(args.optfile) as json_file:
        opts = json.load(json_file)
    

    for k, v in opts.items():
        print(k, v)
    
    # restore from ckpt-1
    opts['file_log'] = False
    opts['restore'] = 1
    g = Gmodel(opts)
    g.solver.save_upred('scipylbfgs')
    # g.solver.predtx('lbfgs', tend, N)

    # # restore from checkpoint 0
    del g
    opts['restore'] = 0
    g = Gmodel(opts)
    g.solver.save_upred('adam')
    # # g.solver.predtx('adam', tend, N)