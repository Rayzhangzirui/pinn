#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
from time import time
import tensorflow_probability as tfp
import os


# Set data type
DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

DIM=1
epsilon =  0.01 # width of diffused domain
T = 300
D = 0.13e-4
rho = 0.025
bound = 0.7 #bound in xy plane

test_case = False

if test_case:
    num_init_train = 100 # initial traning iteration
    n_points = 100 # number of training point
    n_test_points = 100 # number of testing point
    num_hidden_unit = 4 # hidden unit in one layer
    num_add_pt = 10 # number of anchor point
    max_adpt_step = 1 # number of adaptive sampling
    num_adp_train = 1000 #adaptive adam training step
    num_print_every = 10
    num_residual_every = 10
else:
    num_init_train = 20000
    n_points = 10000
    n_test_points=10**(DIM+1)
    num_hidden_unit = 100
    num_add_pt = 1000
    max_adpt_step = 0
    num_adp_train = 1000
    num_print_every = 1000
    num_residual_every = 100


# Set boundary
tmin = 0.
tmax = 1.
xmin = -1.
xmax = 1.

# Draw uniformly sampled collocation points
t_r = tf.random.uniform((n_points,1), 0, 1, dtype=DTYPE)
x_r = tf.random.uniform((n_points,1), -bound, bound, dtype=DTYPE)
X_r = tf.concat([t_r, x_r], axis=1)

def ic(x):
    r = tf.reduce_sum(tf.square(x[:, 1:DIM+1]),1,keepdims=True)
    return 0.1*tf.exp(-1000*r)


# Define model architecture
class PINN_NeuralNet(tf.keras.Model):
    """ Set basic architecture of the PINN model."""

    def __init__(self,
            output_dim=1,
            num_hidden_layers=3, 
            num_neurons_per_layer=100,
            activation='tanh',
            kernel_initializer='glorot_normal',
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
        self.output_transform = lambda x, u: u* x[:, 0:1]+ ic(x)
    
    
    def call(self, X):
        """Forward-pass through neural network."""
        Z = X
        for i in range(self.num_hidden_layers):
            Z = self.hidden[i](Z)
        Z = self.out(Z)
        Z = self.output_transform(X,Z)
        return Z



import scipy.optimize

class PINNSolver():
    def __init__(self, model, X_r):
        self.model = model
        
        # Store collocation points
        self.t = X_r[:,0:1]
        self.x = X_r[:,1:DIM+1]
        
        # Initialize history of losses and global iteration counter
        self.hist = []
        self.iter = 0
    
    def get_r(self):
        
        with tf.GradientTape(persistent=True) as tape:
            # Watch variables representing t and x during this GradientTape
            tape.watch(self.t)
            tape.watch(self.x)
            
            phi = 0.5 + 0.5*tf.tanh((0.5 - tf.sqrt(tf.reduce_sum(tf.square(self.x[:,0:DIM]),1,keepdims=True)))/epsilon)
            
            # Compute current values u(t,x)
            u = self.model(tf.stack([self.t[:,0], self.x[:,0]], axis=1))
            
            u_x = tape.gradient(u, self.x)
            phiux = phi*u_x
            
        u_t = tape.gradient(u, self.t)
        u_xx = tape.gradient(phiux, self.x)
        
        del tape
        return u_t - T*(D*(u_xx) + rho*phi*u*(1-u))
    
    def loss_fn(self, X_data, u_data):
        
        # Compute phi_r
        r = self.get_r()
        phi_r = tf.reduce_mean(tf.square(r))
        
        # Initialize loss
        loss = phi_r
        if X_data is not None:
            # Add phi_0 and phi_b to the loss
            for i in range(len(X_data)):
                u_pred = self.model(X_data[i])
                loss += tf.reduce_mean(tf.square(u_data[i] - u_pred))

        return loss,r
    
    def get_grad(self, X, u):
        with tf.GradientTape(persistent=True) as tape:
            # This tape is for derivatives with
            # respect to trainable variables
            tape.watch(self.model.trainable_variables)
            loss,res = self.loss_fn(X, u)
            
        g = tape.gradient(loss, self.model.trainable_variables)
        del tape
        
        return loss, res, g
    
    
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
            print('It {:05d}: loss = {:10.8e}, maxres = {:10.8e}'.format(self.iter,self.current_loss,np.amax(self.current_res)))
        
        if num_residual_every is not None and self.iter % num_residual_every == 0:
            fname = f"./data{self.iter}.dat"
            u = self.model(tf.concat([self.t,self.x],1))
            data = tf.concat([self.t, self.x, u, self.current_res],1)
            np.savetxt(fname, data.numpy())

        self.hist.append(self.current_loss)
        self.iter+=1

            
        


# Initialize model
model = PINN_NeuralNet()
model.build(input_shape=(None,2))

# Initilize PINN solver
solver = PINNSolver(model, X_r)

# Decide which optimizer should be used
#mode = 'TFoptimizer'
mode = 'ScipyOptimizer'

# Start timer
t0 = time()

# lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000,3000],[1e-2,1e-3,5e-4])
lr = tf.keras.optimizers.schedules.PolynomialDecay(1e-2,decay_steps=num_init_train,end_learning_rate=1e-4)
optim = tf.keras.optimizers.Adam(learning_rate=lr)
solver.solve_with_TFoptimizer(optim, None,None, N=num_init_train)
    

solver.solve_with_ScipyOptimizer(None,None,
                            method='L-BFGS-B',
                            options={'maxiter': num_init_train,
                                     'maxfun': num_init_train,
                                     'maxcor': 50,
                                     'maxls': 50,
                                     'ftol': 1.0*np.finfo(float).eps})

# Print computation time
print('\nComputation time: {} seconds'.format(time()-t0))


model.save('savemodel')
