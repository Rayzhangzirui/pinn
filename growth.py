#!/usr/bin/env python
import sys
sys.path.insert(1, '/home/ziruz16/pinn')


import os
from re import U
from config import *
from pinn import *

os.environ['CUDA_VISIBLE_DEVICES'] = "1"


# paths
model_name = "growth2d_tmp"

tf.random.set_seed(1234)

# hyper parameters
hp = {
    "num_init_train" : 10000, # initial traning iteration
    "n_res_pts" : 20000, # number of residual point
    "n_dat_pts" : 1000, # number of data points
    "num_hidden_layer": 3,
    "num_hidden_unit" : 100, # hidden unit in one layer
    "print_res_every" : 1000, # print residual
    "w_dat" : 0.5, # weight of data, weight of res is 1-w_dat
    "model_name" : model_name,
    "model_dir": './tmp',
    }

lbfgs_opt = {"maxcor": 100, "ftol": 0, "gtol": 0, "maxiter": 10000, "maxls": 50}

# define problem
XDIM = 1
DIM = XDIM+1 # dimension of the problem, including time
EPSILON = 0.01 # width of diffused domain
T = 300.0 # time scale
L = 50.0 # length scale

dcoef = 0.13 # diffusion coeff, un normalized
rho = 0.025

d0 = 0.1 # diffusion coeff, initial guess
rho0 = 0.01 # growth factor, initial guess

D = dcoef*T/(L**2)
RHO = rho*T

BD = 1.1 #normalized bound in xy plane, >1 for diffused domain method

INVERSE = False
inv_dat_file = 'exactu_dim2_n20000.txt'


if INVERSE:
    # if inverse, must provide data
    assert(hp["w_dat"]>0)
    param = {'D':tf.Variable(D, trainable=INVERSE), 'RHO': tf.Variable(RHO, trainable=INVERSE)}
    dat = tf.convert_to_tensor(np.loadtxt(inv_dat_file,delimiter=','),dtype=DTYPE)
    x_dat = dat[:,0:DIM]
    u_dat = dat[:,DIM:DIM+1]
    x_r = x_dat
else:
    # otherwise, sample collocation point
    param = {'D':tf.Variable(D, trainable=INVERSE), 'RHO': tf.Variable(RHO, trainable=INVERSE)}
    x_r = sample_time_space(hp["n_res_pts"], DIM-1, BD, True)
    x_dat = None
    u_dat = None

domain = [[0., 1.]]+[[-BD,BD]]*XDIM
assert(len(domain)==DIM)

def ic(x):
    r2 = tf.reduce_sum(tf.square(x[:, 1:DIM]),1,keepdims=True)
    return 0.1*tf.exp(-500*r2)

def pde(x_r,f):
    # t,x,y normalized here
    t = x_r[:,0:1]
    x = x_r[:,1:2]
    
    list_of_vars = [t,x]
    if DIM > 2:
        y = x_r[:,2:3]
        list_of_vars.append(y)
        if DIM > 3:
            z = x_r[:,3:4]
            list_of_vars.append(z)

    xr = tf.concat(list_of_vars, axis=1)
    u =  f(xr)
    phi = 0.5 + 0.5*tf.tanh((1 - tf.sqrt(tf.reduce_sum(tf.square(xr[:,1:DIM]),1,keepdims=True)))/EPSILON)
    
    u_t = tf.gradients(u, t)[0]
    u_x = tf.gradients(u, x)[0]
    phiux = phi*u_x
    u_xx = tf.gradients(phiux, x)[0]
    lapu = u_xx

    if DIM > 2:
        u_y = tf.gradients(u, y)[0]
        phiuy = phi*u_y
        u_yy = tf.gradients(phiuy, y)[0]
        lapu += u_yy

        if DIM > 3:
            u_z = tf.gradients(u, z)[0]
            phiuz = phi*u_z
            u_zz = tf.gradients(phiuz, z)[0]
            lapu += u_zz
    

    return u_t - (f.param['D']*(lapu) + f.param['RHO']*phi*u*(1-u))

def output_transform(x,u):
    return u* x[:, 0:1]+ ic(x)

u_exact = None

### finish set up, start training

# Initialize model
model = PINN(param=param,
            num_hidden_layers=hp["num_hidden_layer"], 
            num_neurons_per_layer=hp["num_hidden_unit"],
            output_transform=output_transform)
model.build(input_shape=(None,DIM))




# model = tf.keras.models.load_model('/home/ziruz16/pinndata/growth2d_20220810_0852/afteradam')
# Initilize PINN solver
solver = PINNSolver(model, pde,  options=hp)

# Solve with adam
# lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([10000,20000],[1e-2,1e-3,e-4])
# lr = tf.keras.optimizers.schedules.PolynomialDecay(1e-2, decay_steps=hp["num_init_train"], end_learning_rate=1e-5)
lr = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=1e-3, decay_steps=hp["num_init_train"], alpha=1e-3, name=None)
optim = tf.keras.optimizers.Adam(learning_rate=lr)
solver.solve_with_TFoptimizer(optim, x_r, x_dat, u_dat, N=hp["num_init_train"])

# Solve wit bfgs
results = solver.solve_with_ScipyOptimizer(x_r, x_dat, u_dat, method='L-BFGS-B', options=lbfgs_opt)
# results = solver.solve_with_tfbfgs(x_dat,u_dat)


solver.save()