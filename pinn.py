#!/usr/bin/env python
# coding: utf-8
# based on
# https://colab.research.google.com/github/janblechschmidt/PDEsByNNs/blob/main/PINN_Solver.ipynb#scrollTo=zYbXErmCZEkv
import os
from utilgpu import pick_gpu_lowest_memory
os.environ['CUDA_VISIBLE_DEVICES'] = str(pick_gpu_lowest_memory())

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import scipy.optimize
import sys

# make prediction and save
from scipy.io import savemat


from config import *
from util import *

# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class Logger(object):
    def __init__(self,fname):
        self.terminal = sys.stdout
        self.log = open(fname, "a")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

# Define model architecture
class PINN(tf.keras.Model):
    """ Set basic architecture of the PINN model."""

    def __init__(self,
            output_dim=1,
            input_dim=1,
            num_hidden_layers=3, 
            num_neurons_per_layer=100,
            activation='tanh',
            kernel_initializer='glorot_normal',
            output_transform = lambda x,u:u,
            param = None,
            scale = None,
            **kwargs):
        super().__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim
        
        # phyiscal parameters in the model
        self.param = param

        # Define NN architecture
        self.hidden = [tf.keras.layers.Dense(num_neurons_per_layer,
                             activation=tf.keras.activations.get(activation),
                             kernel_initializer=kernel_initializer)
                           for _ in range(self.num_hidden_layers)]
        self.out = tf.keras.layers.Dense(output_dim)
        self.output_transform = output_transform

        self.build(input_shape=(None,input_dim))
        
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
        for i, (shape, weight) in enumerate(zip(self.shapes, weights)):
            self.trainable_variables[i].assign(tf.reshape(weight, shape))

    def to_flat_weights(self, weights):
        """Returns a 1D tf.Tensor representing the `weights`.
        Args:
            weights: A list of tf.Tensor representing the weights.
        Returns:
            A 1D tf.Tensor representing the `weights`.
        """
        return tf.dynamic_stitch(self.indices, weights)


class PINNSolver():
    def __init__(self, model, pde, 
                fdatloss,
                ftestloss,
                xr = None, wr = None,
                xdat = None, udat = None,
                xtest = None, utest=None,
                options=None):
        self.model = model
        
        self.pde = pde
        self.fdatloss = fdatloss
        self.ftestloss = ftestloss

        self.options = options
        
        self.info = {} #empty dictionary to store information

        # set up data
        self.xr =    n2t(xr) # collocation point

        self.xdat =  n2t(xdat) # data point
        self.udat =  n2t(udat) # data value

        self.xtest = n2t(xtest) # test point
        self.utest = n2t(utest) # test value

         # weight of residual
        if wr is None:
            self.wr = tf.ones([tf.shape(xr)[0],1])
        else:
            self.wr = wr

        # Initialize history of losses and global iteration counter
        self.hist = []
        self.iter = 0
        self.header = ''
        self.paramhist = [] # history of trainable model params
        self.current_optimizer = None # current optimizer

        # set up log
        os.makedirs(options['model_dir'], exist_ok=False)
        
        logfile = os.path.join(options['model_dir'],'solver.log')

        # reset stdout https://stackoverflow.com/questions/14245227/python-reset-stdout-to-normal-after-previously-redirecting-it-to-a-file
        sys.stdout = sys.__stdout__ 
        if self.options.get('file_log'):
            # if log to file
            if os.path.exists(logfile): print(f'{logfile} already exist') 
            
            sys.stdout = Logger(logfile)
            

        # set up check point
        self.checkpoint = tf.train.Checkpoint(model)
        self.manager = tf.train.CheckpointManager(self.checkpoint, directory=os.path.join(options['model_dir'],'ckpt'), max_to_keep=4)
        if (not self.manager.latest_checkpoint) and options.get('dicard_ckpt')==False:
            print("Don't restore")
        else:
            if options.get('restore') is not None:
                ckptpath = options.get('restore')
                # assert os.path.exists(ckptpath), f'{ckptpath} not exist'
            else:
                ckptpath = self.manager.latest_checkpoint
            
            self.checkpoint.restore(ckptpath)
            print("Restored from {}".format(ckptpath))

    @tf.function
    def loss_fn(self):
        
        # Compute phi_r
        r = self.pde(self.xr, self.model)
        r2 = tf.math.square(r) * self.wr
            
        loss_res = tf.reduce_mean(r2)
        
        # Initialize loss
        loss_dat = 0.0
        w_dat = self.options.get('w_dat')
        if w_dat > 1e-6:
            # Add phi_0 and phi_b to the loss
            # upred = self.model(self.xdat)
            # loss_dat = tf.math.educe_mean(tf.math.square(self.udat - upred))
            loss_dat = self.fdatloss(self.model, self.xdat)

        loss_tot = loss_res + loss_dat * w_dat

        loss = {'res':loss_res, 'data':loss_dat, 'total':loss_tot}
        return loss
    
    @tf.function
    def get_grad(self):
        """ get loss, residual, gradient
        called by both solve_with_TFoptimizer and solve_with_ScipyOptimizer, need tf.function
        args: x_dat: x data pts, u_dat: value at x.
        """
        with tf.GradientTape(persistent=True) as tape:
            # This tape is for derivatives with
            # respect to trainable variables
            tape.watch(self.model.trainable_variables)
            loss = self.loss_fn()
            
        g = tape.gradient(loss['total'], self.model.trainable_variables)
        del tape

        return loss, g
    
    @tf.function
    def check_exact(self):
        """ check with exact solution if provided
        """
        return self.ftestloss(self.model, self.xtest)
    
    
    def solve_with_TFoptimizer(self, optimizer, N=10000, patience = 1000):
        """This method performs a gradient descent type optimization."""
        
        @tf.function
        def train_step():
            loss, grad_theta = self.get_grad()
            
            # Perform gradient descent step
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss
        
        self.current_optimizer = 'tfadam'
        best_loss = 1e6
        no_improvement_counter = 0
        start = time()
        
        for i in range(N):
            loss = train_step()
            self.current_loss = loss
            self.callback()
            
            # early stopping, 
            if self.current_loss['total'].numpy() < best_loss:
                best_loss = self.current_loss['total']
                no_improvement_counter = 0 # reset counter
            else:
                no_improvement_counter+=1
                if patience is not None and no_improvement_counter == patience:
                    print('No improvement for {} interation'.format(patience))
                    break
        end = time()

        self.info['tfadamiter'] = i
        self.info['tfadamtime'] = (end-start)
        print('adam It:{:05d}, loss {:10.4e}, time {}'.format(i, loss['total'].numpy(), end-start))
        
        self.callback_train_end()


    @timer
    def solve_with_ScipyOptimizer(self, method='L-BFGS-B', **kwargs):
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
            for v in self.model.trainable_variables:
                shape_list.append(v.shape)
                weight_list.extend(v.numpy().flatten())
                
            return weight_list, shape_list

        def set_weight_tensor(weight_list):
            """Function which sets list of weights
            to variables in the model."""
            idx = 0
            for v in self.model.trainable_variables:
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
            loss_dict, grad = self.get_grad()
            
            # Store current loss for callback function            
            loss = loss_dict['total'].numpy().astype(np.float64)
            self.current_loss = loss_dict
            
            # Flatten gradient
            grad_flat = []
            for g in grad:
                grad_flat.extend(g.numpy().flatten())
            
            # Gradient list to array
            grad_flat = np.array(grad_flat,dtype=np.float64)
            
            # Return value and gradient of \phi as tuple
            return loss, grad_flat
        
        self.current_optimizer = 'scipylbfgs'
        x0, shape_list = get_weight_tensor()
        start = time()
        results = scipy.optimize.minimize(fun=get_loss_and_grad,
                                       x0=x0,
                                       jac=True,
                                       method=method,
                                       callback=self.callback,
                                       **kwargs)
        end = time()
        self.info['scipylbfgsiter'] = results.nit
        self.info['scipylbfgstime'] = (end-start)
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult
        print('lbfgs(scipy) It:{:05d}, loss {:10.4e}, time {}, {}.'.format(results.nit, results.fun, end-start, results.message ))
        self.callback_train_end()
        return results

    @timer
    def solve_with_tfbfgs(self,**kwargs):
        
        def bfgs_loss():
            dloss = self.loss_fn()
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

    
    def callback(self,xk=None):
        """ called after one step of iteration in bfgs and adam, 
        scipy.optimize.minimize require first arg to be parameters 
        """
        self.iter+=1
        
        # in the first iteration, create header
        if self.iter == 1:
            trainable_params = []
            # create header
            str_losses = ', '.join('{:<10}'.format(k) for k in self.current_loss) 
            header = '{:<5}, {}'.format('it',str_losses)
            if self.model.param is not None:
                # if not none, add to header
                for pname,ptensor in self.model.param.items():
                    header+= ", {:<10}".format(f'{pname}')
            if self.utest is not None:
                header+= ", {:<10}".format('tmse')
            self.header = header
            print(header)
        
        # write to file if iter==1 or print_res_every
        yeswrite = (self.iter % self.options['print_res_every'] == 0) or (self.iter==1)
        if yeswrite:

            # convert losses to list
            losses = [v.numpy() for _,v in self.current_loss.items()]

            # record data        
            info = [self.iter] + losses

            if self.model.param is not None:
                info.extend( np.array(list(self.model.param.values())))
            
            # if provide test data, output test mse
            if self.utest is not None:
                tmse = self.check_exact()
                info.append(tmse.numpy())    
            info_str = ', '.join('{:10.4e}'.format(k) for k in info[1:])
            print('{:05d}, {}'.format(info[0], info_str))  
            self.hist.append(info)

        # save residual to file with interval save_res_every
        # if self.options['save_res_every'] is not None and self.iter % self.options['save_res_every'] == 0:
        #     fname = f'data{self.iter}.dat'
        #     u = self.model(self.x_r)
        #     data = tf.concat([self.x_r, u],1)
        #     np.savetxt( os.path.join( self.options['model_dir'], fname) , data.numpy())
        
        # save checkpoint
        # if self.options['ckpt_every'] is not None and self.iter % self.options['ckpt_every'] == 0:
        #     save_path = self.manager.save()
        #     print("Saved checkpoint for step {}: {}".format(int(self.iter), save_path))

    def callback_train_end(self):
        if self.options.get('saveckpt'):
            save_path = self.manager.save()
            print("Saved checkpoint for {} step {} {}".format(int(self.iter),self.current_optimizer, save_path))
        else:
            print("checkpoint not saved")

        self.save_upred(self.current_optimizer)

    def save_history(self):
        ''' save training history as txt
        '''
        fpath=os.path.join(self.options['model_dir'],'history.dat')
        hist = np.asarray(self.hist)
        col = hist.shape[1]
        fmt = '%d'+' %.6e'*(col-1) #int for iter, else float
        np.savetxt(fpath, hist, fmt, header = self.header, comments = '')
        print(f'save training hist to {fpath}')

    def save_upred(self,suffix):
        ''' save prediction of u using xr, xtest, xdat
        '''
        savedat = {}
        upredxr = self.model(self.xr)
        residual = self.pde(self.xr, self.model)

        savedat['xr'] = t2n(self.xr)
        savedat['upredxr'] = t2n(upredxr)
        savedat['res'] = t2n(residual)
        
        if self.xdat is not None:
            upredxdat = self.model(self.xdat)
            savedat['xdat'] = t2n(self.xdat)
            savedat['udat'] = t2n(self.udat)
            savedat['upredxdat'] = t2n(upredxdat)

        if self.xtest is not None:
            upredxtest = self.model(self.xtest)
            savedat['xtest'] = t2n(self.xtest)
            savedat['utest'] = t2n(self.utest)
            savedat['upredxtest'] = t2n(upredxtest)

        savedat['rD'] = self.model.param['rD'].numpy()
        savedat['rRHO'] = self.model.param['rRHO'].numpy()

        predfile = os.path.join(self.options['model_dir'],f'upred_{suffix}.mat')
        
        print(f'save upred to {predfile}')
        savemat(predfile,savedat)

    def reweight(self, topk):
        res = np.abs(self.pde(self.xr, self.model).numpy().flatten()) # compute residual
        wres = self.wr
        # get topk idx
        # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        ind = np.argpartition(res, -topk)[-topk:]
        wres[ind]+=1
        self.wr = np.reshape(wres,(-1,1))
    
    def resample(self, topk, xxr):
        ''' compute residual on xxr, add topk points to collocation pts
        '''
        res = np.abs(self.pde(xxr, self.model).numpy().flatten())
        ind = np.argpartition(res, -topk)[-topk:]
        xxr = xxr.numpy()[ind,:]
        newx = tf.convert_to_tensor(xxr)
        self.xr = tf.concat([self.xr, newx],axis=0)
        self.wr = np.ones([tf.shape(self.xr)[0],1])
        return xxr

        
