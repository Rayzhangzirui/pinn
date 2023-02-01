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

tf.keras.backend.set_floatx(DTYPE)

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
            resnet = False,
            **kwargs):
        super().__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.num_neurons_per_layer = num_neurons_per_layer
        self.resnet = resnet

        # phyiscal parameters in the model
        self.param = param

        # Define NN architecture
        self.hidden = [tf.keras.layers.Dense(num_neurons_per_layer,
                             activation=tf.keras.activations.get(activation),
                             kernel_initializer=kernel_initializer)
                           for _ in range(self.num_hidden_layers)]
        self.out = tf.keras.layers.Dense(output_dim)
        self.output_transform = output_transform
        self.paddings = [[0, 0], [0, self.num_neurons_per_layer - self.input_dim]]

        self.build(input_shape=(None,input_dim))
        
    def call(self, X):
        """Forward-pass through neural network."""
        Z = X
        if self.resnet != True:
            for i in range(self.num_hidden_layers):
                Z = self.hidden[i](Z)
        else:
            # resnet implementation
            Z = self.hidden[0](Z)
            for i in range(1,self.num_hidden_layers-1,2):
                Z = self.hidden[i+1](self.hidden[i](Z))+Z
            if i + 2 < self.num_hidden_layers:
                Z = self.hidden[i+2](Z)
            
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
                flosses,
                ftests,
                wr = None,
                xr = None, 
                xdat = None,
                xtest = None,
                options=None):
        self.model = model
        
        self.pde = pde
        
        self.flosses = flosses
        self.ftests = ftests

        self.options = options
        
        self.info = {} #empty dictionary to store information

        # set up data
        self.xr =    (xr).astype(DTYPE) # collocation point
        self.xdat =  (xdat).astype(DTYPE) # data point
        if xtest is not None: 
            self.xtest = (xtest).astype(DTYPE) # test point
        else:
            self.xtest = None
        

         # weight of residual
        if wr is None:
            self.wr = tf.ones([tf.shape(xr)[0],1], dtype = DTYPE)
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
        # self.manager.latest_checkpoint is None if no ckpt found
        
        
        if options['restore'] is not None:
            if isinstance(options['restore'],int):
                # restore check point in the same directory by integer, 0 = ckpt-1
                ckptpath = self.manager.checkpoints[options['restore']]
            else:
                # restore checkpoint by path
                ckptpath = options['restore']
            self.checkpoint.restore(ckptpath)
            print("Restored from {}".format(ckptpath))
        else:
            # try to continue previous simulation
            ckptpath = self.manager.latest_checkpoint
            if ckptpath is not None:
                self.checkpoint.restore(ckptpath)
                print("Restored from {}".format(ckptpath))
            else:
                print("No restore")

        if self.options['trainnnweight'] == False:
            print("do not train NN")
            # self.model.trainable = False
            for l in self.model.layers:
                l.trainable = False
        
        # set layer to be trainable
        if isinstance(self.options['trainnnweight'],int):
            nlayer = self.options['trainnnweight']
            k = 0
            print(f"do not train nn layer <= {nlayer} ")
            # self.model.trainable = False
            for l in self.model.layers:
                if k > nlayer:
                    l.trainable = True
                else:
                    l.trainable = False
                k += 1

        
        
        self.model.summary()


    @tf.function
    def loss_fn(self):
        losses = {}
        total = 0.0
        for key in self.flosses:
            if self.options['weights'].get(key) is not None:
                losses[key] = self.flosses[key](self.model)
                total += losses[key] * self.options['weights'][key]
            
        losses['total'] = total
        return losses
    
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
    
    # @tf.function
    def check_exact(self):
        """ check with exact solution if provided
        """
        testlosses = {}
        for key in self.ftests:
            testlosses[key] = self.ftests[key](self.model)

        return testlosses
    
    
    def solve_with_TFoptimizer(self, optimizer, N=10000, patience = 1000):
        """This method performs a gradient descent type optimization."""

        # @tf.function
        def train_step():
            loss, grad_theta = self.get_grad()
            
            # Perform gradient descent step
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss
        
        self.current_optimizer = self.options['optimizer']
        best_loss = 1e6
        no_improvement_counter = 0
        start = time()
        
        for i in range(N):
            loss = train_step()
            self.current_loss = loss
            self.callback()
            
            # change t
            if self.options.get('randomt') > 0 and i % 10 == 0:
                tend = self.options.get('randomt')
                nrow = self.xr.shape[0]
                self.xr[:,0:1] = np.random.uniform(0, tend, size=(nrow,1))
            
            # randomly set half of the res time to be final time
            if self.options.get('randomtfinal') == True and i % 10 == 0:
                self.xr[:,0:1] = 1.0
                nrow = self.xr.shape[0]
                half = nrow//2
                idx = np.random.choice(nrow, size=half,replace=False)
                self.xr[idx,0:1] = np.random.uniform(size=(half,1))


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
            if self.ftests is not None:
                for key in self.ftests:
                    header+= ", {:<10}".format(key)
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
            if self.ftests is not None:
                test_losses = self.check_exact()
                info +=  [v.numpy() for _,v in test_losses.items()]

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
        # at the end of training of each optimizer, save prediction on xr, xdat
        # also make prediction of xr at various time
        if self.options.get('saveckpt'):
            save_path = self.manager.save()
            print("Saved checkpoint for {} step {} {}".format(int(self.iter),self.current_optimizer, save_path))
        else:
            print("checkpoint not saved")

        self.save_upred(self.current_optimizer)
        self.predtx(self.current_optimizer, 1.0)

    def save_history(self):
        ''' save training history as txt
        '''
        fpath=os.path.join(self.options['model_dir'],'history.dat')
        hist = np.asarray(self.hist)
        col = hist.shape[1]
        fmt = '%d'+' %.6e'*(col-1) #int for iter, else float
        np.savetxt(fpath, hist, fmt, header = self.header, comments = '')
        print(f'save training hist to {fpath}')

    def predtx(self, suffix, tend = 1.0, n = 21):
        # evalute at residual points. not data points. 
        # need Pwm, Pgm , phi etc
        if tend is None:
            tend = 1.0

        savedat = {}
        upredts = [] # prediction at different t
        rests = [] # residual at different t
        
        predfile = os.path.join(self.options['model_dir'],f'upred_txr_{suffix}.mat')
        ts = np.linspace(0, tend, n)
        xr = np.copy(self.xr)
        for t in ts:
            xr[:,0] = t

            upredtxr = self.model(xr)
            restxr = self.pde(xr, self.model)
            
            upredts.append(t2n(upredtxr))
            rests.append(t2n(restxr))
    
        savedat['upredts'] = np.concatenate([*upredts],axis=1)
        savedat['rests'] = np.concatenate([*rests],axis=1)
        savedat['xr'] = xr
        savedat['ts'] = ts
        print(f'save upred of xr at different t to {predfile}')
        savemat(predfile,savedat)
        
                

    def save_upred(self,suffix):
        ''' save prediction of u using xr, xtest, xdat
        '''
        savedat = {}

        upredxr = self.model(self.xr)
        savedat['xr'] = t2n(self.xr)
        savedat['upredxr'] = t2n(upredxr)

        resxr = self.pde(self.xr, self.model)
        savedat['resxr'] = t2n(resxr)
        
        
        if self.xdat is not None:
            upredxdat = self.model(self.xdat)
            savedat['xdat'] = t2n(self.xdat)
            savedat['upredxdat'] = t2n(upredxdat)

        # can not evaluate residual at xtest, need Pwm Pwg
        if self.xtest is not None:
            upredxtest = self.model(self.xtest)
            savedat['xtest'] = t2n(self.xtest)
            savedat['upredxtest'] = t2n(upredxtest)

        for key in self.model.param:
            savedat[key] = self.model.param[key].numpy()

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

        
