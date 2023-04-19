#!/usr/bin/env python

import sys
sys.path.insert(1, '/home/ziruz16/pinn')

from config import *
from glioma import *
from options import Options

tf.random.set_seed(1)



if __name__ == "__main__":
    
    args = sys.argv[1:]

    optobj = Options()
    optobj.parse_args(*sys.argv[1:])

    optobj.opts['model_dir'] = str_from_dict(optobj.opts, optobj.opts['tag'], [])


    g = Gmodel(optobj.opts)
    g.solve()

    # successfule finish
    f = open("commands.txt", "a")
    f.write(' '.join(sys.argv))
    f.write('\n')
    f.write('finish\n')
    f.close()


