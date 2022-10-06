#!/usr/bin/env python
# rescale rho and D by characteristic rho and D

import sys
sys.path.insert(1, '/home/ziruz16/pinn')

from config import *
from glioma import *

tf.random.set_seed(1)

# hyper parameters
lbfgs_opt = {"maxcor": 100, "ftol": 0, "gtol": 0,'maxfun': 10000, "maxiter": 10000, "maxls": 50}

opts = {
    "tag" : 'different number fo training points',
    "model_dir": 'tmp',
    "num_init_train" : 100000, # initial traning iteration
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
    'inv_dat_file': '',
    "inverse":True,
    "lbfgs_opts":lbfgs_opt,
    }

files = [
# 'dat_dim2_dw0.13_rho0.025_ix164_iy116_iz99_tend150_n20000_nzinterp_add.mat',
# 'dat_dim2_dw0.13_rho0.025_ix164_iy116_iz99_tend150_n20000_nzinterp_mult.mat',
# 'dat_dim2_dw0.13_rho0.025_ix164_iy116_iz99_tend150_n20000_nzinterp_add_cor.mat',
# 'dat_dim2_dw0.13_rho0.025_ix164_iy116_iz99_tend150_n20000_interpnz_add.mat',
'dat_dim2_dw0.13_rho0.025_ix164_iy116_iz99_tend150_n20000_none.mat'
]

tags = [
# 'nzadd',
# 'nzmult',
# 'nzaddcor'
# 'nzinterpadd'
'nznone'
]


# test
# opts['num_init_train'] = 200
# lbfgs_opt['maxiter'] = 2

for i in range(len(files)):
    opts['inv_dat_file'] = files[i]
    opts['model_dir'] = f'{tags[i]}'
    g = Gmodel(opts)
    g.solve()
    del g
