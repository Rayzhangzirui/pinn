#!/usr/bin/env python
# rescale rho and D by characteristic rho and D

import sys
sys.path.insert(1, '/home/ziruz16/pinn')

from config import *
from glioma import *


# hyper parameters
lbfgs_opt = {"maxcor": 100, "ftol": 0, "gtol": 0,'maxfun': 20000, "maxiter": 20000, "maxls": 50}

opts = {
    "tag" : 'different random initial',
    "model_dir": 'tmp',
    "num_init_train" : 80000, # initial traning iteration
    "n_res_pts" : 20000, # number of residual point
    "num_hidden_layer": 3,
    "num_hidden_unit" : 100, # hidden unit in one layer
    "print_res_every" : 100, # print residual
    "save_res_every" : None, # save residual
    "w_dat" : 1, # weight of data, weight of res is 1
    "ckpt_every": 20000,
    "patience":1000,
    "file_log":True,
    "saveckpt":True,
    "inverse":True,
    "lbfgs_opts":lbfgs_opt,
    'inv_dat_file': '/home/ziruz16/pinn/data/dat_dim2_dw0.13_rho0.025_ix164_iy116_iz99_tend150.mat'
    }

for i in range(2,3):
    opts['model_dir'] = f's{i}'
    tf.random.set_seed(i)
    g = Gmodel(opts)
    g.solve()
    del g
