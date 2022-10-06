#!/usr/bin/env python
# gradient enhanced model

import sys
sys.path.insert(1, '/home/ziruz16/pinn')
from burgers import *



# paths
hp["model_name"] = "plain2"
hp["model_dir"] = "./plain2"
adamstep = 100000
patience = 1000
lbfgs_opt['maxiter'] = 10000

# Initialize model

# Initilize PINN solver
solver = PINNSolver(model, pde, xr = xr, xtest = xtest, utest = utest, options=hp)

lr = 1e-3
optim = tf.keras.optimizers.Adam(learning_rate=lr)
solver.solve_with_TFoptimizer(optim, N=adamstep, patience = patience)
results = solver.solve_with_ScipyOptimizer(method='L-BFGS-B', options=lbfgs_opt)

solver.save(header='t x u w')
