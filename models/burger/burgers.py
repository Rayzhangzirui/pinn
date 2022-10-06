#!/usr/bin/env python
# definition of burgers equation

import sys
sys.path.insert(1, '/home/ziruz16/pinn')

import os
from config import *
from pinn import *

# paths
tf.random.set_seed(1)

# define problem
XDIM = 1
DIM = XDIM + 1 # dimension of the problem, including time

BD = 1
domain = [[0., 1.],[-BD,BD]]
assert(len(domain)==DIM)

def ic(x):
    return -tf.math.sin(np.pi*x)

nu = 0.01/np.pi

# dudt + u dudx = nu d2ud2x, x=[-1,1],t=[0,1]
# u(x,0) = -sin(pi x), u(-1,t) = u(1,t) = 0
@tf.function
def pde(xr,f):
    # t,x,y normalized here
    t = xr[:,0:1]
    x = xr[:,1:2]
    
    xr = tf.concat([t,x], axis=1)
    u =  f(xr)

    u_t = tf.gradients(u, t)[0]
    u_x = tf.gradients(u, x)[0]
    u_xx = tf.gradients(u_x, x)[0]

    return u_t + u * u_x - nu * u_xx


def output_transform(x,u):
    return ((x[:, 1:2]**2-1)*u* x[:, 0:1]+ ic(x[:, 1:2]))


# hyper parameters
hp = {
    "num_init_train" : 10000, # initial traning iteration
    "n_res_pts" : 2000, # number of residual point
    "n_dat_pts" : 1000, # number of data points
    "num_hidden_layer": 3,
    "num_hidden_unit" : 32, # hidden unit in one layer
    "print_res_every" : 100, # print residual
    "save_res_every" : None, # save residual
    "w_dat" : 1, # weight of data, weight of res is 1
    "ckpt_every": 20000,
    "patience":1000
    }

lbfgs_opt = {"maxcor": 100, "ftol": 0, "gtol": 0,'maxfun': 20000, "maxiter": 10000, "maxls": 50}



# sample
xr = sample(hp["n_res_pts"], domain)
xdat = None
udat = None
testdatfile = 'burgers_gd_n10201.txt'
dat = tf.convert_to_tensor(np.loadtxt(testdatfile,delimiter=','),dtype=DTYPE)
xtest = dat[:,0:XDIM+1]
utest = dat[:,-1:]

model = PINN(
            num_hidden_layers=hp["num_hidden_layer"], 
            num_neurons_per_layer=hp["num_hidden_unit"],
            output_transform=output_transform)
model.build(input_shape=(None,DIM))