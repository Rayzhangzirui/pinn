#!/usr/bin/env python
# gradient enhanced model

import sys
sys.path.insert(1, '/home/ziruz16/pinn')

import os
from re import U
from config import *
from pinn import *

os.environ['CUDA_VISIBLE_DEVICES'] = "1"


# paths
model_name = "growth2d_inv_xgrad"

tf.random.set_seed(1234)

# hyper parameters
hp = {
    "num_init_train" : 70000, # initial traning iteration
    "n_res_pts" : 20000, # number of residual point
    "n_dat_pts" : 1000, # number of data points
    "num_hidden_layer": 3,
    "num_hidden_unit" : 100, # hidden unit in one layer
    "print_res_every" : 100, # print residual
    "save_res_every" : None, # save residual
    "w_dat" : 1, # weight of data, weight of res is 1
    "w_xr": 1e-2,
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

d0 = 0.1 # diffusion coeff, initial guess
rho0 = 0.01 # growth factor, initial guess

D = dcoef*T/(L**2)
RHO = rho*T

#normalized bound in xy plane, >1 for diffused domain method
# look at pde solver, solution is 0 outside 0.8
BD = 1.1

INVERSE = True
inv_dat_file = 'exactu_dim2_n20000.txt'

if INVERSE:
    # if inverse, must provide data
    assert(hp["w_dat"]>0)
    param = {'D':tf.Variable(D*2, trainable=INVERSE), 'RHO': tf.Variable(RHO*2, trainable=INVERSE)}
    dat = tf.convert_to_tensor(np.loadtxt(inv_dat_file,delimiter=','),dtype=DTYPE)
    x_dat = dat[:,0:DIM]
    u_dat = dat[:,DIM:DIM+1]
    x_r = x_dat
else:
    # otherwise, sample collocation point
    hp["w_dat"] = 0
    param = {'D':tf.Variable(D, trainable=INVERSE), 'RHO': tf.Variable(RHO, trainable=INVERSE)}
    x_r = sample_time_space(hp["n_res_pts"], DIM-1, BD, True)
    x_dat = None
    u_dat = None

domain = [[0., 1.],[-BD,BD],[-BD,BD]]
assert(len(domain)==DIM)

def ic(x):
    r2 = tf.reduce_sum(tf.square(x[:, 1:DIM]),1,keepdims=True)
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

    res = u_t - (f.param['D']*(phiux_x + phiuy_y) + f.param['RHO']*phi*u*(1-u))

    # enhanced by dt
    rest = tf.gradients(res,t)[0]
    resx = tf.gradients(res,x)[0]
    resy = tf.gradients(res,y)[0]
    dres = rest+resx+resy
    return res, dres

def output_transform(x,u):
    return u* x[:, 0:1]+ ic(x)
    
class PINNSolverXgrad(PINNSolver):
    @tf.function
    def loss_fn(self, x_r, x_dat = None, u_dat = None, w_r=None):
        
        def wl2norm(r,w_r):
            # return the weighted l2 norm
            r2 = tf.math.square(r)
            if w_r is not None:
                r2 = r2*w_r
            return tf.reduce_mean(r2)

        # Compute phi_r
        r,dr = self.pde(x_r, self.model)
        loss_res = wl2norm(r,w_r)
        loss_dres = wl2norm(dr,w_r)

        # Initialize loss
        loss_dat = 0.
        if x_dat is not None:
            # Add phi_0 and phi_b to the loss
            u_pred = self.model(x_dat)
            loss_dat = tf.reduce_mean(tf.square(u_dat - u_pred)) *  self.options['w_dat']

        loss_tot = loss_res + loss_dat* self.options['w_dat'] + loss_dres* self.options['w_xr']

        loss = {'res':loss_res, 'data':loss_dat, 'total':loss_tot, 'dres':loss_dres}


        maxres = tf.math.reduce_max(tf.math.abs(r))
        return loss, maxres

### finish set up, start training



# Initialize model
model = PINN(param=param,
            num_hidden_layers=hp["num_hidden_layer"], 
            num_neurons_per_layer=hp["num_hidden_unit"],
            output_transform=output_transform)
model.build(input_shape=(None,DIM))


# model = tf.keras.models.load_model('/home/ziruz16/pinndata/growth2d_20220810_0852/afteradam')
# Initilize PINN solver
solver = PINNSolverXgrad(model, pde, options=hp)

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