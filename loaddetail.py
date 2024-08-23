#!/usr/bin/env python
# load model and make prediction

import sys
sys.path.insert(1, '/home/ziruz16/pinn')

from config import *
from glioma import *
import json

if __name__ == "__main__":
    
    optfile = sys.argv[1] #option file, used to load model
    
    # Opening JSON file
    with open(optfile) as json_file:
        opts = json.load(json_file)

    opts['file_log'] = None    

    for k, v in opts.items():
        print(k, v)
    
    # restore from ckpt-1
    opts['restore'] = 1
    g = Gmodel(opts)
    xrpred, A, b = g.model.lastlayer(g.dataset.X_res)

    xdatpred, _, _ = g.model.lastlayer(g.dataset.X_dat)

    savedat = {}
    savedat['xrpred'] = [t2n(v) for v in xrpred]
    savedat['xdatpred'] = [t2n(v) for v in xdatpred]
    savedat['A'] = t2n(A)
    savedat['b'] = t2n(b)

    predfile = os.path.join(opts['model_dir'],f'detail_last_layer.mat')
    print(f'save upred to {predfile}')
    savemat(predfile,savedat)
    