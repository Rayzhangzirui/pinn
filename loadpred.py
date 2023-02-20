#!/usr/bin/env python
# load model and make prediction

import sys
sys.path.insert(1, '/home/ziruz16/pinn')

from config import *
from glioma import *
import json

if __name__ == "__main__":
    
    optfile = sys.argv[1] #option file, used to load model
    tend = eval(sys.argv[2]) #final time
    N = eval(sys.argv[3]) # interval
 
    # Opening JSON file
    with open(optfile) as json_file:
        opts = json.load(json_file)
    

    for k, v in opts.items():
        print(k, v)
    
    # restore from ckpt-1
    opts['file_log'] = False
    opts['restore'] = 1
    g = Gmodel(opts)
    g.solver.save_upred('scipylbfgs')
    # g.solver.predtx('lbfgs', tend, N)

    # restore from checkpoint 0
    del g
    opts['restore'] = 0
    g = Gmodel(opts)
    g.solver.save_upred('tfadam')
    # g.solver.predtx('adam', tend, N)