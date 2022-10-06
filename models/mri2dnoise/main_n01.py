#!/usr/bin/env python
# rescale rho and D by characteristic rho and D

import sys
sys.path.insert(1, '/home/ziruz16/pinn')

from config import *
from pinn import *

# paths
model_name = "mri2dn01"

tf.random.set_seed(1)

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

# define problem
XDIM = 2
DIM = XDIM + 1 # dimension of the problem, including time

tfinal = 150.0 # final time [day]
bound = 21.1108 # domain boundary[mm]
Dw_actual = 0.13
rho_actual = 0.025
Dw_char = 0.1 # characteristic diffusion coeff, white matter, [mm^2/day]
rho_char = 0.02 # characteristic proliferation, [1/day]
sigma = 0.1 # noise level

# prepare data
n = hp["n_res_pts"]
inv_dat_file = '/home/ziruz16/pinn/data/datmri_dim2_n20000_st1_spline.txt'
xdat, udat, phi, pwm, pgm = read_mri_dat(n,inv_dat_file,DIM)

# make sure that the space location of xr and xdat are the same
test_dat_file = '/home/ziruz16/pinn/data/datmri_dim2_n20000_st0_spline.txt'
xtest, utest, _, _, _ = read_mri_dat(n,test_dat_file,DIM)

# determine length scale
rdat = tf.math.sqrt(tf.reduce_sum(tf.square(xdat[:, 1:DIM]),1,keepdims=True))
bound = tf.reduce_max(rdat).numpy() # domain boundary[mm], maximum radius

# choose scale
L = bound
T = L/np.sqrt(Dw_char*rho_char)
print(f'T = {T}, L = {L}')
scale = tf.constant([T,L,L],dtype=DTYPE)

# scale input
xdat = xdat/scale
xtest = xtest/scale

# define residual points
xr = xtest

# add noise
udat = udat + tf.random.normal([n,1], 0, sigma, DTYPE)

# nondimensional parameters
Dw = np.sqrt(Dw_char/rho_char)/L
Dg = Dw/10
RHO = np.sqrt(rho_char/Dw_char)*L


INVERSE = True
Dphi = (pwm * Dw + pgm * Dg) * phi
param = {'rD':tf.Variable(2.0, trainable=INVERSE), 'rRHO': tf.Variable(2.0, trainable=INVERSE)}


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
    
    u_t = tf.gradients(u, t)[0]

    u_x = tf.gradients(u, x)[0]
    u_xx = tf.gradients(Dphi*u_x, x)[0]
    
    u_y = tf.gradients(u, y)[0]
    u_yy = tf.gradients(Dphi*u_y, y)[0]

    
    res = u_t - (f.param['rD']*(u_xx + u_yy) + f.param['rRHO']*RHO*phi*u*(1-u))
    return res

def output_transform(x,u):
    return u* x[:, 0:1]+ ic(x)
    

### finish set up, start training
# Initialize model
model = PINN(param=param,
            input_dim=DIM,
            scale = scale,
            num_hidden_layers=hp["num_hidden_layer"], 
            num_neurons_per_layer=hp["num_hidden_unit"],
            output_transform=output_transform)

# Initilize PINN solver
solver = PINNSolver(model, pde, xr = xr, xdat = xdat, udat = udat,
                    xtest=xtest,utest=utest, options=hp)

# Solve with adam
optim = tf.keras.optimizers.Adam(learning_rate=1e-3)
if __name__ == "__main__":
    solver.solve_with_TFoptimizer(optim, N=hp["num_init_train"],patience = hp["patience"])
    results = solver.solve_with_ScipyOptimizer(method='L-BFGS-B', options=lbfgs_opt)
    solver.save(header = 't x y u w')