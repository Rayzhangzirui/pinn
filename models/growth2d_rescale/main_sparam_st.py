#!/usr/bin/env python
# single time inference

import sys
sys.path.insert(1, '/home/ziruz16/pinn')

from config import *
from pinn import *

# paths
model_name = "scale_param_st"

tf.random.set_seed(1234)

# hyper parameters
hp = {
    "num_init_train" : 100000, # initial traning iteration
    "n_res_pts" : 20000, # number of residual point
    "num_hidden_layer": 3,
    "num_hidden_unit" : 100, # hidden unit in one layer
    "print_res_every" : 100, # print residual
    "save_res_every" : None, # save residual
    "w_dat" : 1, # weight of data, weight of res is 1
    "model_name" : model_name,
    "model_dir": './'+model_name,
    "ckpt_every": 20000,
    "patience":1000
    }

lbfgs_opt = {"maxcor": 100, "ftol": 0, "gtol": 0,'maxfun': 20000, "maxiter": 20000, "maxls": 50}

# test
# hp["num_init_train"] = 1000
# lbfgs_opt["maxiter"] = 1000

# define problem
XDIM = 2
DIM = XDIM + 1 # dimension of the problem, including time

tfinal = 300.0 # final time [day]
bound = 50.0 # domain boundary[mm]
dcoef = 0.13 # diffusion coeff, white matter, [mm^2/day]
rho = 0.025 # proliferation, [1/day]
ddmwidth = 5 # width of diffsed domain [mm]
bd = bound + ddmwidth * 2

# choose length scale
T = 1/rho
L = np.math.sqrt(dcoef/rho)
D = 1
RHO = 1


#
TF = tfinal/T # final time, unitless
BD = bd/L # domain bound, unitless
EPSILON = ddmwidth/L # ddm width, unitless


# Generate data, data point only final time, need residual point for all time
N = 20000
xr = sample_time_space(N, XDIM, BD, spherical=True,tfinal=TF)

INVERSE = True
inv_dat_file = '/home/ziruz16/pinn/data/exactu_dim2_n20000_unscale_tfinal.txt'
param = {'rD':tf.Variable(2.0, trainable=INVERSE), 'rRHO': tf.Variable(2.0, trainable=INVERSE)}
dat = tf.convert_to_tensor(np.loadtxt(inv_dat_file,delimiter=','),dtype=DTYPE)
xdat = dat[:,0:DIM]/tf.constant([T,L,L],dtype=DTYPE)
udat = dat[:,DIM:DIM+1]



domain = [[0., TF],[-BD,BD],[-BD,BD]]
assert(len(domain)==DIM)

def ic(x):
    r2 = tf.reduce_sum(tf.square(x[:, 1:DIM]),1,keepdims=True)
    return 0.1*tf.exp(-0.1*r2*(L**2))


def pde(x_r,f):
    # t,x,y normalized here
    t = x_r[:,0:1]
    x = x_r[:,1:2]
    y = x_r[:,2:3]
    xr = tf.concat([t,x,y], axis=1)
    u =  f(xr)
    phi = 0.5 + 0.5*tf.tanh((BD - tf.sqrt(tf.reduce_sum(tf.square(xr[:,1:DIM]),1,keepdims=True)))/EPSILON)
    
    u_t = tf.gradients(u, t)[0]

    u_x = tf.gradients(u, x)[0]
    phiux_x = tf.gradients(phi*u_x, x)[0]
    
    u_y = tf.gradients(u, y)[0]
    phiuy_y = tf.gradients(phi*u_y, y)[0]

    res = u_t - (f.param['rD'] * D *(phiux_x + phiuy_y) + f.param['rRHO']*RHO*phi*u*(1-u))
    return res

def output_transform(x,u):
    return u* x[:, 0:1]+ ic(x)
    

### finish set up, start training
# Initialize model
model = PINN(param=param,
            input_dim=DIM,
            scale = [T,L,L],
            num_hidden_layers=hp["num_hidden_layer"], 
            num_neurons_per_layer=hp["num_hidden_unit"],
            output_transform=output_transform)

# Initilize PINN solver
solver = PINNSolver(model, pde, xr = xr, xdat = xdat, udat = udat, options=hp)
optim = tf.keras.optimizers.Adam(learning_rate=1e-3)
# Solve with adam
if __name__ == "__main__":
    solver.solve_with_TFoptimizer(optim, N=hp["num_init_train"],patience = hp["patience"])
    results = solver.solve_with_ScipyOptimizer(method='L-BFGS-B', options=lbfgs_opt)
    solver.save(header = 't x y u w')