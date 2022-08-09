#!/usr/bin/env python
import os
from re import U
from config import *
from pinn import *

import sys

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

DIM = 1 # dimension of the problem
epsilon = 0.01 # width of diffused domain
T = 300
D = 0.13e-4
rho = 0.025
bound = 0.7 #bound in xy plane

test_case = False
inverse = True
# tf.config.run_functions_eagerly(True)

# paths
model_name = "test"
timetag = datetime.now().strftime('%Y%m%d_%H%M')
dirname = f"{model_name}_{timetag}"
MD_SAVE_DIR = os.path.join(DATADIR,dirname)
# mkdir if not exist
if not test_case:
    os.makedirs(MD_SAVE_DIR,exist_ok=True)

tf.random.set_seed(1234)

# hyper parameters
hp = {
    "num_init_train" : 1000, # initial traning iteration
    "n_res_pts" : 100, # number of residual point
    "n_dat_pts" : 1000, # number of data points
    "n_test_points" : 100, # number of testing point
    "num_hidden_unit" : 4, # hidden unit in one layer
    "num_add_pt" : 10, # number of anchor point
    "max_adpt_step" : 1, # number of adaptive sampling
    "num_adp_train" : 1000, #adaptive adam training step
    "print_res_every" : 100, # print residual
    "save_res_every" : None, # save residual
    "w_dat" : 0.5, # weight of data, weight of res is 1-w_dat
    "model_name" : model_name,
    "model_dir": MD_SAVE_DIR,
    }

lbfgs_opt = {"maxcor": 100, "ftol": 0, "gtol": 1e-8, "maxiter": 10000, "maxls": 50}

## 1d test case
param = tf.Variable([1., 0.5], trainable=True)
param_true = tf.Variable([2., 1.0], trainable=False)
if inverse is False:
    # param.assign(param_true)
    param = param_true

DIM=1 
domain = [[0., 1.]]

def output_transform(x,u):
    return u

def pde(x_r, f):
    x = x_r
    u =  f(x)
    
    u_x = tf.gradients(u, x)[0]
    # u_xx = tf.gradients(u_x, x)[0]

    return u_x-(f.param[0]*x+f.param[1])

def u_exact(x):
    return param_true[0]*x**2/2 + param_true[1]*x

# Draw uniformly sampled collocation points
x_r = sample(hp["n_res_pts"], domain)

x_dat = None
u_dat = None
if hp["w_dat"] > 0:
    x_dat = x_r
    if u_exact is not None:
        u_dat = u_exact(x_dat)
    # u_dat = interpsol('sol1d.txt', 100, 100, x_dat)


# Initialize model
model = PINN(param=param,
            num_hidden_layers=2, 
            num_neurons_per_layer=hp["num_hidden_unit"])
model.build(input_shape=(None,DIM))

# Initilize PINN solver
solver = PINNSolver(model, pde, x_r, x_dat=x_r, u_dat=u_dat, u_exact=u_exact, options=hp)

# Start timer
t0 = time()

# Solve with adam
# lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000,3000],[1e-2,1e-3,5e-4])
lr = tf.keras.optimizers.schedules.PolynomialDecay(1e-2, decay_steps=hp["num_init_train"], end_learning_rate=1e-4)
optim = tf.keras.optimizers.Adam(learning_rate=lr)
solver.solve_with_TFoptimizer(optim, x_dat, u_dat, N=hp["num_init_train"])
    
# Solve wit bfgs
results = solver.solve_with_ScipyOptimizer(x_dat, u_dat, method='L-BFGS-B', options=lbfgs_opt)
# results = solver.solve_with_tfbfgs(x_dat,u_dat)

# Print computation time
print('\nComputation time: {} seconds'.format(time()-t0))

if not test_case:
    model.save(os.path.join(MD_SAVE_DIR,'savemodel'))
    solver.save_history(os.path.join(MD_SAVE_DIR, 'history.dat'))