#!/usr/bin/env python
import os
from re import U
from config import *
from pinn import *
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


# paths
model_name = "growth2d"
timetag = datetime.now().strftime('%Y%m%d_%H%M')
dirname = f"{model_name}_{timetag}"
MD_SAVE_DIR = os.path.join(DATADIR,dirname)
# mkdir if not exist
os.makedirs(MD_SAVE_DIR,exist_ok=True)

tf.random.set_seed(1234)

# hyper parameters
hp = {
    "num_init_train" : 40000, # initial traning iteration
    "n_res_pts" : 20000, # number of residual point
    "n_dat_pts" : 1000, # number of data points
    "n_test_points" : 100, # number of testing point
    "num_hidden_layer": 3,
    "num_hidden_unit" : 100, # hidden unit in one layer
    "num_add_pt" : 10, # number of anchor point
    "max_adpt_step" : 1, # number of adaptive sampling
    "num_adp_train" : 1000, #adaptive adam training step
    "print_res_every" : 1000, # print residual
    "save_res_every" : None, # save residual
    "w_dat" : 0, # weight of data, weight of res is 1-w_dat
    "model_name" : model_name,
    "model_dir": MD_SAVE_DIR,
    }

lbfgs_opt = {"maxcor": 100, "ftol": 0, "gtol": 1e-8, "maxiter": 10000, "maxls": 50}

# define problem
DIM = 3 # dimension of the problem, including time
epsilon = 0.01 # width of diffused domain
T = 300
D = 0.13e-4
rho = 0.025
bound = 0.55 #bound in xy plane

inverse = False

## 1d test case
param = tf.Variable([D, rho], trainable=True)
param_true = tf.Variable([D, rho], trainable=False)
if inverse is False:
    # param.assign(param_true)
    param = param_true

domain = [[0., 1.],[-bound,bound],[-bound,bound]]
assert(len(domain)==DIM)

def ic(x):
    r = tf.reduce_sum(tf.square(x[:, 1:DIM]),1,keepdims=True)
    return 0.1*tf.exp(-1000*r)

def pde(x_r,f):
    t = x_r[:,0:1]
    x = x_r[:,1:2]
    y = x_r[:,2:3]
    xr = tf.concat([t, x, y], axis=1)
    u =  f(xr)
    phi = 0.5 + 0.5*tf.tanh((0.5 - tf.sqrt(tf.reduce_sum(tf.square(xr[:,1:DIM]),1,keepdims=True)))/epsilon)
    
    u_x = tf.gradients(u, x)[0]
    u_y = tf.gradients(u, y)[0]
    phiux = phi*u_x
    phiuy = phi*u_y
    
    u_t = tf.gradients(u, t)[0]
    u_xx = tf.gradients(phiux, x)[0]
    u_yy = tf.gradients(phiuy, y)[0]
    
    return u_t - T*(D*(u_xx+u_yy) + rho*phi*u*(1-u))

def output_transform(x,u):
    return u* x[:, 0:1]+ ic(x)

u_exact = None

# Draw uniformly sampled collocation points
# x_r = sample(hp["n_res_pts"], domain)
x_r = sample_time_space(hp["n_res_pts"], DIM-1, bound, True)


x_dat = None
u_dat = None
if hp["w_dat"] > 0:
    x_dat = x_r
    if u_exact is not None:
        u_dat = u_exact(x_dat)
    # u_dat = interpsol('sol1d.txt', 100, 100, x_dat)


# Initialize model
model = PINN(param=param,
            num_hidden_layers=hp["num_hidden_layer"], 
            num_neurons_per_layer=hp["num_hidden_unit"],
            output_transform=output_transform)
model.build(input_shape=(None,DIM))


# model = tf.keras.models.load_model('/home/ziruz16/pinndata/growth2d_20220810_0852/afteradam')
# Initilize PINN solver
solver = PINNSolver(model, pde, x_r, x_dat=x_r, u_dat=u_dat, u_exact=u_exact, options=hp)

# Start timer
t0 = time()

# Solve with adam
# lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([10000,20000],[1e-2,1e-3,e-4])
# lr = tf.keras.optimizers.schedules.PolynomialDecay(1e-2, decay_steps=hp["num_init_train"], end_learning_rate=1e-5)
lr = 1e-4
optim = tf.keras.optimizers.Adam(learning_rate=lr)
solver.solve_with_TFoptimizer(optim, x_dat, u_dat, N=hp["num_init_train"])

model.save(os.path.join(MD_SAVE_DIR,'afteradam'))

# Solve wit bfgs
results = solver.solve_with_ScipyOptimizer(x_dat, u_dat, method='L-BFGS-B', options=lbfgs_opt)
# results = solver.solve_with_tfbfgs(x_dat,u_dat)

# Print computation time
print('\nComputation time: {} seconds'.format(time()-t0))


model.save(os.path.join(MD_SAVE_DIR,'savemodel'))
solver.save_history(os.path.join(MD_SAVE_DIR, 'history.dat'))