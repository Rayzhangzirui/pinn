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
    parser.add_argument("-n", "--num", type=int, default=-1, help="int of ckpt file")

    args = parser.parse_args()
    # Opening JSON file
    with open(args.optfile) as json_file:
        opts = json.load(json_file)
    

    for k, v in opts.items():
        print(k, v)
    
    # restore from ckpt-1
    opts['file_log'] = False
    opts['restore'] = ''
    g = Gmodel(opts)

    # list of ckpt to restore
    if args.num<0:
        ns = [0,1]
    else:
        ns = [args.num]
    
    # restore and predict
    for n in ns:
        if n==0:
            optimizer = opts['optimizer']
        else:
            optimizer = 'scipylbfgs'
        
        g.model.checkpoint.restore(g.model.manager.checkpoints[args.num])
        g.solver.save_upred(optimizer)
        g.solver.predtx(optimizer)