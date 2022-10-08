#!/usr/bin/env python
# rescale rho and D by characteristic rho and D

import sys
sys.path.insert(1, '/home/ziruz16/pinn')

from config import *
from glioma import *

tf.random.set_seed(1)

# hyper parameters
lbfgs_opt = None

opts = {
    "tag" : 'different variance of noise',
    "model_dir": './',
    "num_init_train" : 100, # initial traning iteration
    "n_res_pts" : None,
    "num_hidden_layer": 3,
    "num_hidden_unit" : 64, # hidden unit in one layer
    "print_res_every" : 100, # print residual
    "save_res_every" : None, # save residual
    "w_dat" : 1, # weight of data, weight of res is 1
    "ckpt_every": 20000,
    "patience":1000,
    "file_log":True,
    "saveckpt":True,
    "inverse":True,
    "lbfgs_opts":lbfgs_opt,
    'inv_dat_file': 'dat_dim2_dw0.13_rho0.025_ix164_iy116_iz99_tend150_n5000_std.mat'
    }

g = Gmodel(opts)
g.solve()
