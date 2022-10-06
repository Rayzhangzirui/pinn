#!/usr/bin/env python
# definition for tumor growth model

import sys
sys.path.insert(1, '/home/ziruz16/pinn')

from config import *
from pinn import *

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
    "ckpt_every": 20000,
    "patience":1000
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
BD = 1.05

domain = [[0., 1.],[-BD,BD],[-BD,BD]]
assert(len(domain)==DIM)

def ic(x):
    r2 = tf.math.square(x[:,1:2]) + tf.math.square(x[:,2:3])
    return 0.1*tf.exp(-500*r2)

@tf.function
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

    res = u_t - (f.param['rD'] * D *(phiux_x + phiuy_y) + f.param['rRHO']*RHO*phi*u*(1-u))
    return res

def output_transform(x,u):
    return u* x[:, 0:1]+ ic(x)


INVERSE = False
inv_dat_file = 'exactu_dim2_n20000.txt'

param = {'rD':tf.Variable(1.0, trainable=INVERSE),
         'rRHO': tf.Variable(1.0, trainable=INVERSE),
         }
dat = tf.convert_to_tensor(np.loadtxt(inv_dat_file,delimiter=','),dtype=DTYPE)
xtest = dat[:,0:DIM]
utest = dat[:,DIM:DIM+1]

