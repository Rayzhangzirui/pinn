#!/usr/bin/env python
# burgers equation

import sys
sys.path.insert(1, '/home/ziruz16/pinn')

import os
from burgers import *



# paths
hp["model_name"] = "rar2"
hp["model_dir"] = "./rar2"
lbfgs_opt['maxiter'] = 10000


# Initilize PINN solver
solver = PINNSolver(model, pde, xr = xr, xtest = xtest, utest = utest, options=hp)

lr = 1e-3
optim = tf.keras.optimizers.Adam(learning_rate=lr)
# solver.solve_with_TFoptimizer(optim, N=hp["num_init_train"])

newxs = []
solver.solve_with_TFoptimizer(optim, N=20000, patience = 1000)
topk = 10
nrar = 40
nrarpt = 100000
for i in range(nrar):
    solver.solve_with_TFoptimizer(optim, N=20000, patience = 1000)
    xxr = sample(nrarpt, domain)
    newx = solver.resample(topk,xxr)
    newxs.extend(newx)

solver.solve_with_ScipyOptimizer(method='L-BFGS-B', options=lbfgs_opt)

solver.save(header = 't x u w')

# save added residual points
np.savetxt( os.path.join(hp['model_dir'],'./newx.txt'), newxs)