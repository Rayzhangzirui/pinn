#!/usr/bin/env python
# coding: utf-8
import math
import tensorflow as tf
import numpy as np
from time import time
import tensorflow_probability as tfp
import scipy.optimize

from datetime import datetime
from config import *
from util import *




# domain = [[0., 1.],[-0.7,0.7]]

# def ic(x):
#     r = tf.reduce_sum(tf.square(x[:, 1:DIM]),1,keepdims=True)
#     return 0.1*tf.exp(-1000*r)

# def pde(x_r,f):
#     t = x_r[:,0:1]
#     x = x_r[:,1:DIM]
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

# u_exact = None


# Define model architecture
class PINN_NeuralNet(tf.keras.Model):
    """ Set basic architecture of the PINN model."""

    def __init__(self,
            output_dim=1,
            num_hidden_layers=3, 
            num_neurons_per_layer=100,
            activation='tanh',
            kernel_initializer='glorot_normal',
            output_transform = lambda x,u:u,
            train_param = None,
            **kwargs):
        super().__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim
        
        self.train_param = train_param

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

# copy from 
# https://github.com/lululxvi/deepxde/blob/9f0d86dea2230478d8735615e2ad518c62efe6e2/deepxde/optimizers/tensorflow/tfp_optimizer.py#L103
class LossAndFlatGradient(object):
    """A helper class to create a function required by tfp.optimizer.lbfgs_minimize.
    Args:
        trainable_variables: Trainable variables.
        build_loss: A function to build the loss function expression.
    """

    def __init__(self, trainable_variables, build_loss):
        self.trainable_variables = trainable_variables
        self.build_loss = build_loss

        # Shapes of all trainable parameters
        self.shapes = tf.shape_n(trainable_variables)
        self.n_tensors = len(self.shapes)

        # Information for tf.dynamic_stitch and tf.dynamic_partition later
        count = 0
        self.indices = []  # stitch indices
        self.partitions = []  # partition indices
        for i, shape in enumerate(self.shapes):
            n = np.product(shape)
            self.indices.append(
                tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape)
            )
            self.partitions.extend([i] * n)
            count += n
        self.partitions = tf.constant(self.partitions)

    @tf.function
    def __call__(self, weights_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.
        Args:
           weights_1d: a 1D tf.Tensor.
        Returns:
            A scalar loss and the gradients w.r.t. the `weights_1d`.
        """
        # Set the weights
        self.set_flat_weights(weights_1d)
        with tf.GradientTape() as tape:
            # Calculate the loss
            loss = self.build_loss()
        # Calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss, self.trainable_variables)
        grads = tf.dynamic_stitch(self.indices, grads)
        return loss, grads

    def set_flat_weights(self, weights_1d):
        """Sets the weights with a 1D tf.Tensor.
        Args:
            weights_1d: a 1D tf.Tensor representing the trainable variables.
        """
        weights = tf.dynamic_partition(weights_1d, self.partitions, self.n_tensors)
        for i, (shape, param) in enumerate(zip(self.shapes, weights)):
            self.trainable_variables[i].assign(tf.reshape(param, shape))

    def to_flat_weights(self, weights):
        """Returns a 1D tf.Tensor representing the `weights`.
        Args:
            weights: A list of tf.Tensor representing the weights.
        Returns:
            A 1D tf.Tensor representing the `weights`.
        """
        return tf.dynamic_stitch(self.indices, weights)


    

class PINNSolver():
    def __init__(self, model, pde, x_r, x_dat=None, u_dat=None, u_exact = None, options=None, lbfgs_opt = None):
        self.model = model
        
        # Store collocation points
        self.x_r = x_r
        
        self.pde = pde

        # if exact sol is provided
        self.u_exact = u_exact
        
        self.options = options

        
        # Initialize history of losses and global iteration counter
        self.w_dat = options["w_dat"]
        self.hist = []
        self.iter = 0

    @tf.function
    def loss_fn(self, x_dat, u_dat):
        
        # Compute phi_r
        r = self.pde(self.x_r, self.model)
        loss_res = tf.reduce_mean(tf.square(r))
        
        # Initialize loss
        if x_dat is not None:
            # Add phi_0 and phi_b to the loss
            u_pred = self.model(x_dat)
            loss_dat = tf.reduce_mean(tf.square(u_dat - u_pred)) * self.w_dat
        else:
            loss_dat = 0.

        loss_tot = loss_res * (1-self.w_dat) + loss_dat*self.w_dat

        loss = {'res':loss_res, 'data':loss_dat, 'total':loss_tot}
        return loss, r
    
    
    @tf.function
    def get_grad(self, x_dat, u_dat):
        """ get loss, residual, gradient
        called by both solve_with_TFoptimizer and solve_with_ScipyOptimizer, need tf.function
        args: x_dat: x data pts, u_dat: value at x.
        """
        with tf.GradientTape(persistent=True) as tape:
            # This tape is for derivatives with
            # respect to trainable variables
            tape.watch(self.model.trainable_variables)
            loss,res = self.loss_fn(x_dat, u_dat)
            
        g = tape.gradient(loss['total'], self.model.trainable_variables)
        del tape

        return loss, res, g
    

    def check_exact(self):
        """ check with exact solution if provided
        """
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
            self.current_loss = loss
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
            loss_dict, res, grad = self.get_grad(X, u)
            
            # Store current loss for callback function            
            loss = loss_dict['total'].numpy().astype(np.float64)
            self.current_loss = loss_dict
            self.current_res = res
            
            # Flatten gradient
            grad_flat = []
            for g in grad:
                grad_flat.extend(g.numpy().flatten())
            
            # Gradient list to array
            grad_flat = np.array(grad_flat,dtype=np.float64)
            
            # Return value and gradient of \phi as tuple
            return loss, grad_flat
        
        
        results = scipy.optimize.minimize(fun=get_loss_and_grad,
                                       x0=x0,
                                       jac=True,
                                       method=method,
                                       callback=self.callback,
                                       **kwargs)
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult
        it=results.nit
        loss_final = results.fun
        print('bfgs(scipy) It:{:05d}, loss {:10.4e}'.format(it, loss_final))
        return results

    
    
    def solve_with_tfbfgs(self, X, u_dat, **kwargs):
        
        @tf.function
        def bfgs_loss():
            dloss,_=self.loss_fn(X, u_dat)
            return dloss['total']


        func = LossAndFlatGradient(self.model.trainable_variables, bfgs_loss)
        initial_position = func.to_flat_weights(self.model.trainable_variables)
        
        results = tfp.optimizer.lbfgs_minimize(
            func,
            initial_position,
            previous_optimizer_results=None,
            **kwargs
        )
        
        loss_final = results.objective_value.numpy()
        it = results.num_iterations.numpy()
        print('bfgs(tfp) It:{:05d}, loss {:10.4e} '.format(it, loss_final))

        return results

        
        
    def callback(self,x=None):
        """ called after bfgs and adam, 
        scipy.optimize.minimize require first arg to be parameters 
        """
        
        if self.iter % self.options["print_res_every"] == 0:
            str_loss = '[{:10.4e}, {:10.4e}, {:10.4e}] '.format(self.current_loss['res'],self.current_loss['data'],self.current_loss['total'])
            # if exact solution is provided, 
            if self.u_exact:
                mse,maxe = self.check_exact()
                str_metric = '[{:10.4e}, {:10.4e}]'.format(mse.numpy(),maxe.numpy())
                str_loss += str_metric

            
            print('It {:05d}: {} {:10.4e}'.format(self.iter, str_loss ,np.amax(np.abs(self.current_res))))


        # save residual to file with interval save_res_every
        if self.options["save_res_every"] is not None and self.iter % self.options["save_res_every"] == 0:
            fname = f"data{self.iter}.dat"
            u = self.model(self.x_r)
            data = tf.concat([self.x_r, u, self.current_res],1)
            np.savetxt( os.path.join( self.options["model_dir"], fname) , data.numpy())

        # loss history
        self.hist.append(self.current_loss)
        self.iter+=1

            
        

