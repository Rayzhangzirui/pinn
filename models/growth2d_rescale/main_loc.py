#!/usr/bin/env python
# Copy from main_rescale.py, also inter location

import sys
sys.path.insert(1, '/home/ziruz16/pinn')

import os
from re import U
from config import *
from pinn import *

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# paths
model_name = "growth2d_inv_rescale_loc"

tf.random.set_seed(1234)

# hyper parameters
hp = {
    "num_init_train" : 100000, # initial traning iteration
    "n_res_pts" : 20000, # number of residual point
    "n_dat_pts" : 1000, # number of data points
    "num_hidden_layer": 3,
    "num_hidden_unit" : 100, # hidden unit in one layer
    "print_res_every" : 100, # print residual
    "save_res_every" : None, # save residual
    "w_dat" : 1, # weight of data, weight of res is 1
    "model_name" : model_name,
    "model_dir": './'+model_name,
    "ckpt_every": 20000,
    "require_improvement":None
    }

lbfgs_opt = {"maxcor": 100, "ftol": 0, "gtol": 0,'maxfun': 20000, "maxiter": 20000, "maxls": 50}

# define problem
XDIM = 2
DIM = XDIM + 1 # dimension of the problem, including time
EPSILON = 0.01 # width of diffused domain
T = 300.0 # time scale
L = 50.0 # length scale

dcoef = 0.13 # diffusion coeff, un normalized
rho = 0.025

D = dcoef*T/(L**2)
RHO = rho*T

#normalized bound in xy plane, >1 for diffused domain method
# look at pde solver, solution is 0 outside 0.8
BD = 1.1

INVERSE = True
inv_dat_file = 'exactu_dim2_n20000.txt'

assert(hp["w_dat"]>0)
param = {'rD':tf.Variable(2.0, trainable=INVERSE),
         'rRHO': tf.Variable(2.0, trainable=INVERSE),
         'x0': tf.Variable(0.2, trainable=INVERSE),
         'y0': tf.Variable(0.2, trainable=INVERSE)
         }
dat = tf.convert_to_tensor(np.loadtxt(inv_dat_file,delimiter=','),dtype=DTYPE)
x_dat = dat[:,0:DIM]
u_dat = dat[:,DIM:DIM+1]
x_r = x_dat

domain = [[0., 1.],[-BD,BD],[-BD,BD]]
assert(len(domain)==DIM)

def ic(x):
    r2 = tf.math.square(x[:,1:2] - param['x0']) + tf.math.square(x[:,2:3] - param['y0'])
    return 0.1*tf.exp(-500*r2)


def pde(x_r,f):
    # t,x,y normalized here
    t = x_r[:,0:1]
    x = x_r[:,1:2]
    y = x_r[:,2:3]

    xr = tf.concat([t,x,y], axis=1)
    u =  f(xr)
    phi = 0.5 + 0.5*tf.tanh((1 - tf.sqrt(tf.reduce_sum(tf.square(xr[:,1:DIM]),1,keepdims=True)))/EPSILON)
    
    u_t = tf.gradients(u, t)[0]

    u_x = tf.gradients(u, x)[0]
    phiux_x = tf.gradients(phi*u_x, x)[0]
    
    u_y = tf.gradients(u, y)[0]
    phiuy_y = tf.gradients(phi*u_y, y)[0]

    res = u_t - (param['rD'] * D *(phiux_x + phiuy_y) + param['rRHO']*RHO*phi*u*(1-u))
    return res

def output_transform(x,u):
    return u* x[:, 0:1]+ ic(x)
    
model = PINN(param=param,
            num_hidden_layers=hp["num_hidden_layer"], 
            num_neurons_per_layer=hp["num_hidden_unit"],
            output_transform=output_transform)
model.build(input_shape=(None,DIM))


# model = tf.keras.models.load_model('/home/ziruz16/pinndata/growth2d_20220810_0852/afteradam')
# Initilize PINN solver
solver = PINNSolver(model, pde, options=hp)

# Solve with adam
# lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([10000,20000],[1e-2,1e-3,e-4])
# lr = tf.keras.optimizers.schedules.PolynomialDecay(1e-2, decay_steps=hp["num_init_train"], end_learning_rate=1e-5)
# lr = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=1e-3, decay_steps=hp["num_init_train"], alpha=1e-3, name=None)
optim = tf.keras.optimizers.Adam(learning_rate=1e-3)

solver.solve_with_TFoptimizer(optim, x_r, x_dat, u_dat, N=hp["num_init_train"])

# Solve wit bfgs
results = solver.solve_with_ScipyOptimizer(x_r, x_dat, u_dat, method='L-BFGS-B', options=lbfgs_opt)

solver.save()

# make prediction
upxr = model(x_r)
np.savetxt( os.path.join(hp['model_dir'],'./predxr.txt'), upxr)