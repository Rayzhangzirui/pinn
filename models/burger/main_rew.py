#!/usr/bin/env python
# burgers equation

import sys
sys.path.insert(1, '/home/ziruz16/pinn')

import os
from burgers import *



# paths
hp["model_name"] = "rew"
hp["model_dir"] = "./rew"
lbfgs_opt['maxiter'] = 10000


# Initilize PINN solver
solver = PINNSolver(model, pde, xr = xr, xtest = xtest, utest = utest, options=hp)

lr = 1e-3
optim = tf.keras.optimizers.Adam(learning_rate=lr)

topk = 10
for i in range(40):
    solver.solve_with_TFoptimizer(optim, N=20000, patience = 1000)
    solver.reweight(topk)

solver.solve_with_ScipyOptimizer(method='L-BFGS-B', options=lbfgs_opt)

solver.save(header = 't x u w')
