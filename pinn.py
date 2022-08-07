#!/usr/bin/env python
# coding: utf-8
from config import *
import tensorflow as tf
import numpy as np
from time import time
# import tensorflow_probability as tfp
import scipy.optimize
import os
from datetime import datetime
from util import *

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

DIM=1 # dimension of the problem
epsilon = 0.01 # width of diffused domain
T = 300
D = 0.13e-4
rho = 0.025
bound = 0.7 #bound in xy plane

test_case = True


# paths
model_name = "test"
timetag = datetime.now().isoformat(timespec='minutes')
dirname = f"{timetag}{model_name}"
MD_SAVE_DIR = os.path.join(DATADIR,dirname)
if not test_case:
    os.makedirs(MD_SAVE_DIR,exist_ok=False)

if test_case:
    tf.random.set_seed(1234)
    num_init_train = 100 # initial traning iteration
    n_res_pts = 100 # number of residual point
    n_dat_pts = 100 # number of data points
    n_test_points = 100 # number of testing point
    num_hidden_unit = 4 # hidden unit in one layer
    num_add_pt = 10 # number of anchor point
    max_adpt_step = 1 # number of adaptive sampling
    num_adp_train = 1000 #adaptive adam training step
    num_print_every = 10 
    num_residual_every = None 
    w_dat = 0 # weight of data, weight of res is 1-w_dat
    model_name = "test"
    
else:
    num_init_train = 20000
    n_res_pts = 10000
    n_dat_pts = 10000
    n_test_points=10**(DIM+1)
    num_hidden_unit = 100
    num_add_pt = 1000
    max_adpt_step = 0
    num_adp_train = 1000
    num_print_every = 1000
    num_residual_every = 100
    w_dat = 0.5




# def ic(x):
#     r = tf.reduce_sum(tf.square(x[:, 1:DIM+1]),1,keepdims=True)
#     return 0.1*tf.exp(-1000*r)

# def pde(x_r,f):
#     t = x_r[:,0:1]
#     x = x_r[:,1:DIM+1]
#     xr = tf.concat([t, x], axis=1)
#     u =  f(xr)
#     phi = 0.5 + 0.5*tf.tanh((0.5 - tf.sqrt(tf.reduce_sum(tf.square(x),1,keepdims=True)))/epsilon)
    
#     u_x = tf.gradients(u, x)[0]
#     phiux = phi*u_x
    
#     u_t = tf.gradients(u, t)[0]
#     u_xx = tf.gradients(phiux, x)[0]
    
#     return u_t - T*(D*(u_xx) + rho*phi*u*(1-u))

# def output_transform(x,u):
#     return u* x[:, 0:1]+ ic(x)



domain = [[0., 1.]]

def output_transform(x,u):
    return x * (1-x) * u


def pde(x_r, f):
    x = x_r[:,0:1]
    u =  f(x)
    
    u_x = tf.gradients(u, x)[0]
    u_xx = tf.gradients(u_x, x)[0]
    
    return u_xx-2

def u_exact(x):
    return (x-1)*x



# Draw uniformly sampled collocation points
x_r = sample(n_res_pts, domain)

x_dat = None
u_dat = None
if w_dat > 0:
    # x_dat = sample(n_dat_pts, DIM, bound, False)
    x_dat = x_r
    u_dat = interpsol('sol1d.txt', 100, 100, x_dat)



# Define model architecture
class PINN_NeuralNet(tf.keras.Model):
    """ Set basic architecture of the PINN model."""

    def __init__(self,
            output_dim=1,
            num_hidden_layers=3, 
            num_neurons_per_layer=100,
            activation='tanh',
            kernel_initializer='glorot_normal',
            output_transform = output_transform,
            **kwargs):
        super().__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim
        

        # Define NN architecture
        self.hidden = [tf.keras.layers.Dense(num_neurons_per_layer,
                             activation=tf.keras.activations.get(activation),
                             kernel_initializer=kernel_initializer)
                           for _ in range(self.num_hidden_layers)]
        self.out = tf.keras.layers.Dense(output_dim)
        # self.output_transform = tf.keras.layers.Lambda(lambda x, u: u* x[:, 0:1]+ ic(x))
        self.output_transform = output_transform
        
    
    
    def call(self, X):
        """Forward-pass through neural network."""
        Z = X
        for i in range(self.num_hidden_layers):
            Z = self.hidden[i](Z)
        Z = self.out(Z)
        Z = self.output_transform(X,Z)
        return Z



    

class PINNSolver():
    def __init__(self, model, x_r, w_dat, u_exact = None):
        self.model = model
        
        # Store collocation points
        self.x_r = x_r

        # if exact sol is provided
        self.u_exact = u_exact
        
        
        # Initialize history of losses and global iteration counter
        self.w_dat = w_dat
        self.hist = []
        self.iter = 0

    def loss_fn(self, x_dat, u_dat):
        
        # Compute phi_r
        r = pde(self.x_r, self.model)
        phi_r = tf.reduce_mean(tf.square(r)) * (1 - self.w_dat)
        
        # Initialize loss
        loss = phi_r * (1-self.w_dat)
        if x_dat is not None:
            # Add phi_0 and phi_b to the loss
            u_pred = self.model(x_dat)
            loss += tf.reduce_mean(tf.square(u_dat - u_pred)) * self.w_dat
        return loss, r, 
    
    @tf.function
    def get_grad(self, X, u):
        with tf.GradientTape(persistent=True) as tape:
            # This tape is for derivatives with
            # respect to trainable variables
            tape.watch(self.model.trainable_variables)
            loss,res = self.loss_fn(X, u)
            
        g = tape.gradient(loss, self.model.trainable_variables)
        del tape
        
        return loss, res, g
    

    def check_exact(self):
        # check with exact solution if provided
        assert(self.u_exact is not None)
        up = self.model(self.x_r)
        ue = self.u_exact(self.x_r)
        mse = tf.math.reduce_mean((up-ue)**2)
        maxe = tf.math.reduce_max(tf.math.abs(up-ue))
        return mse, maxe
    
    def solve_with_TFoptimizer(self, optimizer, X, u, N=1001):
        """This method performs a gradient descent type optimization."""
        
        @tf.function
        def train_step():
            loss, res, grad_theta = self.get_grad(X, u)
            
            # Perform gradient descent step
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss, res
        
        for i in range(N):
            
            loss,res = train_step()
            self.current_loss = loss.numpy()
            self.current_res = res.numpy()
            self.callback()

    def solve_with_ScipyOptimizer(self, X, u, method='L-BFGS-B', **kwargs):
        """This method provides an interface to solve the learning problem
        using a routine from scipy.optimize.minimize.
        (Tensorflow 1.xx had an interface implemented, which is not longer
        supported in Tensorflow 2.xx.)
        Type conversion is necessary since scipy-routines are written in Fortran
        which requires 64-bit floats instead of 32-bit floats."""
        
        def get_weight_tensor():
            """Function to return current variables of the model
            as 1d tensor as well as corresponding shapes as lists."""
            
            weight_list = []
            shape_list = []
            
            # Loop over all variables, i.e. weight matrices, bias vectors and unknown parameters
            for v in self.model.variables:
                shape_list.append(v.shape)
                weight_list.extend(v.numpy().flatten())
                
            weight_list = tf.convert_to_tensor(weight_list)
            return weight_list, shape_list

        x0, shape_list = get_weight_tensor()
        
        def set_weight_tensor(weight_list):
            """Function which sets list of weights
            to variables in the model."""
            idx = 0
            for v in self.model.variables:
                vs = v.shape
                
                # Weight matrices
                if len(vs) == 2:  
                    sw = vs[0]*vs[1]
                    new_val = tf.reshape(weight_list[idx:idx+sw],(vs[0],vs[1]))
                    idx += sw
                
                # Bias vectors
                elif len(vs) == 1:
                    new_val = weight_list[idx:idx+vs[0]]
                    idx += vs[0]
                    
                # Variables (in case of parameter identification setting)
                elif len(vs) == 0:
                    new_val = weight_list[idx]
                    idx += 1
                    
                # Assign variables (Casting necessary since scipy requires float64 type)
                v.assign(tf.cast(new_val, DTYPE))
        
        def get_loss_and_grad(w):
            """Function that provides current loss and gradient
            w.r.t the trainable variables as vector. This is mandatory
            for the LBFGS minimizer from scipy."""
            
            # Update weights in model
            set_weight_tensor(w)
            # Determine value of \phi and gradient w.r.t. \theta at w
            loss, res, grad = self.get_grad(X, u)
            
            # Store current loss for callback function            
            loss = loss.numpy().astype(np.float64)
            self.current_loss = loss
            self.current_res = res
            
            # Flatten gradient
            grad_flat = []
            for g in grad:
                grad_flat.extend(g.numpy().flatten())
            
            # Gradient list to array
            grad_flat = np.array(grad_flat,dtype=np.float64)
            
            # Return value and gradient of \phi as tuple
            return loss, grad_flat
        
        
        return scipy.optimize.minimize(fun=get_loss_and_grad,
                                       x0=x0,
                                       jac=True,
                                       method=method,
                                       callback=self.callback,
                                       **kwargs)
        
        
    def callback(self, xr=None):
        
        if self.iter % num_print_every == 0:
            print('It {:05d}: loss = {:10.4e}, maxres = {:10.4e}'.format(self.iter,self.current_loss,np.amax(self.current_res)))
            # if exact solution is provided, 
            if self.u_exact:
                mse,maxe = self.check_exact()
                print('mse = {:10.4e}, maxe = {:10.4e}'.format(mse.numpy(),maxe.numpy()))


        
        if num_residual_every is not None and self.iter % num_residual_every == 0:
            fname = f"data{self.iter}.dat"
            u = self.model(tf.concat([self.t,self.x],1))
            data = tf.concat([self.t, self.x, u, self.current_res],1)
            np.savetxt( os.path.join(MD_SAVE_DIR,fname) , data.numpy())

        

        self.hist.append(self.current_loss)
        self.iter+=1

            
        


# Initialize model
model = PINN_NeuralNet()
model.build(input_shape=(None,DIM))

# Initilize PINN solver
solver = PINNSolver(model, x_r, w_dat, u_exact)

# Start timer
t0 = time()

# lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000,3000],[1e-2,1e-3,5e-4])
lr = tf.keras.optimizers.schedules.PolynomialDecay(1e-2,decay_steps=num_init_train,end_learning_rate=1e-4)
optim = tf.keras.optimizers.Adam(learning_rate=lr)
solver.solve_with_TFoptimizer(optim, x_dat,u_dat, N=num_init_train)
    

solver.solve_with_ScipyOptimizer(x_dat,u_dat,
                            method='L-BFGS-B',
                            options={'maxiter': num_init_train,
                                     'maxfun': num_init_train,
                                     'maxcor': 50,
                                     'maxls': 50,
                                     'ftol': 1.0*np.finfo(float).eps})

# Print computation time
print('\nComputation time: {} seconds'.format(time()-t0))



if not test_case:
    model.save(os.path.join(MD_SAVE_DIR,'model_name'))
