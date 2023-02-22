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

from weight import Weighting
from rbf_for_tf2.rbflayer import RBFLayer
from tflbfgs import LossAndFlatGradient

tf.keras.backend.set_floatx(DTYPE)

glob_trainable_variables = []


# debug, not compatible with tf.gradients
# tf.config.run_functions_eagerly(True) 
# RuntimeError: tf.gradients is not supported when eager execution is enabled. Use tf.GradientTape instead.

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
            regularizer = None,
            userff = False,
            userbf = False,
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
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer= regularizer)
                           for _ in range(self.num_hidden_layers)]
        
        if userff == True:
            rbf =  tf.keras.layers.experimental.RandomFourierFeatures(
                output_dim=num_neurons_per_layer,
                scale=1.,
                trainable = True,
                kernel_initializer='gaussian')
            self.hidden =   [rbf] + self.hidden
        
        if userbf == True:
            rbf =  RBFLayer(num_neurons_per_layer, betas=1.)
            self.hidden =   [rbf] + self.hidden



        self.out = tf.keras.layers.Dense(output_dim)
        self.output_transform = output_transform
        self.paddings = [[0, 0], [0, self.num_neurons_per_layer - self.input_dim]]

        self.build(input_shape=(None,input_dim))
        
    def call(self, X):
        """Forward-pass through neural network."""
        Z = X

        if self.resnet != True:
            for i in range(len(self.hidden)):
                Z = self.hidden[i](Z)
        else:
            # resnet implementation
            Z = self.hidden[0](Z)
            for i in range(1,len(self.hidden)-1,2):
                Z = self.hidden[i+1](self.hidden[i](Z))+Z
            if i + 2 < len(self.hidden):
                Z = self.hidden[i+2](Z)
            
        Z = self.out(Z)
        Z = self.output_transform(X,Z)
        return Z
    
    def lastlayer(self,X):
        '''output last layer output, weights, bias'''
        if self.resnet == True:
            sys.exit('intermediate output not for resnet')
        Z = X
        Zout = []
        for i in range(len(self.hidden)):
                Z = self.hidden[i](Z)
                Zout.append(Z)
        return Zout, self.out.weights[0], self.out.weights[1]
    
    def set_trainable_layer(self, arg):
        '''Set trainable property of each layer
        '''
        if arg == False:
            print("do not train NN")
            # self.model.trainable = False
            for l in self.layers:
                l.trainable = False
        
        # set layer to be trainable
        if isinstance(arg,int):
            nlayer = arg
            k = 0
            print(f"do not train nn layer <= {nlayer} ")
            # self.trainable = False
            for l in self.layers:
                if k > nlayer:
                    l.trainable = True
                else:
                    l.trainable = False
                k += 1


class PINNSolver():
    def __init__(self, model, pde, 
                flosses,
                ftests,
                geomodel=None, 
                wr = None,
                xr = None, 
                xdat = None,
                xtest = None,
                options=None):
        self.model = model
        self.geomodel = geomodel
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
        
        # dynamic weighting
        self.weighting = Weighting(self.options['weights'], **self.options['weightopt'])

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
        self.gradlog = None #log for gradient

        # set up log
        os.makedirs(options['model_dir'], exist_ok=True)
        
        logfile = os.path.join(options['model_dir'],'solver.log')

        # reset stdout https://stackoverflow.com/questions/14245227/python-reset-stdout-to-normal-after-previously-redirecting-it-to-a-file
        sys.stdout = sys.__stdout__ 
        if self.options.get('file_log'):
            # if log to file
            if os.path.exists(logfile): 
                print(f'{logfile} already exist') 
                sys.exit()
            
            sys.stdout = Logger(logfile)
        else:
            print('skip file log')
        
        self.manager = self.setup_ckpt(self.model, ckptdir = 'ckpt', restore = self.options['restore'])
        if self.geomodel is not None:
            self.managergeo = self.setup_ckpt(self.geomodel, ckptdir = 'geockpt', restore = self.options['restore'])

        # may set some layer trainable = False
        self.model.set_trainable_layer(self.options.get('trainweight'))

        self.model.summary()

        global glob_trainable_variables
        glob_trainable_variables+=  self.model.trainable_variables
        if self.geomodel is not None:
            glob_trainable_variables +=  self.geomodel.trainable_variables

    def setup_ckpt(self, model, ckptdir = 'ckpt', restore = None):
         # set up check point
        checkpoint = tf.train.Checkpoint(model)
        manager = tf.train.CheckpointManager(checkpoint, directory=os.path.join(self.options['model_dir'],ckptdir), max_to_keep=4)
        # manager.latest_checkpoint is None if no ckpt found

        if self.options['restore'] is not None:
            # self.options['restore'] is a number or a path
            if isinstance(self.options['restore'],int):
                # restore check point in the same directory by integer, 0 = ckpt-1
                ckptpath = manager.checkpoints[self.options['restore']]
            else:
                # restore checkpoint by path
                ckptpath = os.path.join(self.options['restore'],ckptdir,'ckpt-2')
            checkpoint.restore(ckptpath)
            print("Restored from {}".format(ckptpath))
        else:
            # self.options['restore'] is None
            # try to continue previous simulation
            ckptpath = manager.latest_checkpoint
            if ckptpath is not None:
                checkpoint.restore(ckptpath)
                print("Restored from {}".format(ckptpath))
            else:
                # if no previous ckpt
                print("No restore")

        return manager 

    @tf.function
    def loss_fn(self):
        uwlosses = {}
        total = 0.0
        for key in self.weighting.weight_keys:
            uwlosses[key] = self.flosses[key]()
            total += self.weighting.alphas[key] * uwlosses[key]

        uwlosses['total'] = total
        return uwlosses
    
    @tf.function
    def get_grad(self):
        """ get loss, residual, gradient
        called by both solve_with_TFoptimizer and solve_with_ScipyOptimizer, need tf.function
        """
        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
            # This tape is for derivatives with
            # respect to trainable variables
            # tape.watch(self.model.trainable_variables)
            tape.watch(glob_trainable_variables)
            loss = self.loss_fn()
            
        g = tape.gradient(loss['total'], glob_trainable_variables)
        del tape

        return loss, g
    
    # @tf.function
    def get_grad_by_loss(self):
        """ get gradient by each loss
        to study gradient, look at weighted loss
        """
        
        wlosses = {}
        grad = {}
        total = 0.0
        with tf.GradientTape(persistent=True,  watch_accessed_variables=True) as tape:
            # This tape is for derivatives with
            # respect to trainable variables
            # tape.watch(self.model.trainable_variables)
            tape.watch(glob_trainable_variables)
            for key in self.weighting.weight_keys:
                wlosses[key] = self.weighting.alphas[key]* self.flosses[key]()
                total += wlosses[key]
        wlosses['total'] = total

        for key in self.weighting.weight_keys:
            grad[key] = tape.gradient(wlosses[key], glob_trainable_variables)
        
        grad['total'] = tape.gradient(wlosses['total'], glob_trainable_variables)

        del tape
        
        return wlosses, grad
    
    @tf.function
    def check_exact(self):
        """ check with exact solution if provided
        """
        testlosses = {}
        for key in self.ftests:
            testlosses[key] = self.ftests[key](self.model)

        return testlosses
    
    
    def solve_with_TFoptimizer(self, optimizer, N=10000, patience = 1000):
        """This method performs a gradient descent type optimization."""

        @tf.function
        def train_step():
            loss, grad_theta = self.get_grad()
            
            # Perform gradient descent step
            optimizer.apply_gradients(zip(grad_theta, glob_trainable_variables))
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
            for v in glob_trainable_variables:
                shape_list.append(v.shape)
                weight_list.extend(v.numpy().flatten())
                
            return weight_list, shape_list

        def set_weight_tensor(weight_list):
            """Function which sets list of weights
            to variables in the model."""
            idx = 0
            for v in glob_trainable_variables:
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


        func = LossAndFlatGradient(glob_trainable_variables, bfgs_loss)
        initial_position = func.to_flat_weights(glob_trainable_variables)
        
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

    def process_grad(self, graddict):
        '''grad is dict, key = loss name, value = list of gradient, param, weight, bias, etc'''
        if self.iter == 1:
            fpath=os.path.join(self.options['model_dir'],'grad.log')
            self.gradlog = open(fpath, "w")
            self.gradlog.write('iter, ')
            for losskey in graddict:
                for i,v in enumerate(glob_trainable_variables):
                    self.gradlog.write(losskey + '/' + v.name.removesuffix(':0') + ',')
            self.gradlog.write('\n')
        

        self.gradlog.write(f'{self.iter},')
        for losskey, gradlist in graddict.items():
            
            for grad in gradlist:
                if grad is None:
                    g = None
                else:
                    g = tf.reduce_sum(tf.square(grad)).numpy()
                self.gradlog.write(f'{g},')
        
        self.gradlog.write('\n')


    def process_pde(self, pdedict):
        ''' ouput individual terms of the pde as .mat file. named by iteration number
        '''
        pdedict = {k: t2n(v) for k, v in pdedict.items()}
        pdedict['xr'] = self.xr
        
        dirpath = os.path.join(self.options['model_dir'], 'pde')
        if self.iter == 1:
            os.makedirs(dirpath, exist_ok=True)

        predfile = os.path.join(dirpath,f'pde_{self.iter}.mat')
        savemat(predfile,pdedict)
        return None


    def callback(self,xk=None):
        """ called after one step of iteration in bfgs and adam, 
        scipy.optimize.minimize require first arg to be parameters 
        """
        self.iter+=1
        self.weighting.update_weights(self.current_loss)
        

        # in the first iteration, create header
        if self.iter == 1:
            trainable_params = []
            # create header
            str_losses = ', '.join('{:<10}'.format(k) for k in self.current_loss)
            header = '{:<5}, {:<10}'.format('it',str_losses)
            if self.weighting.method != 'constant':
                str_weight = ', '.join('W{:<10}'.format(k) for k in self.weighting.weight_keys) 
                header += ', {}'.format(str_weight)
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

            if self.options['gradnorm'] == True:
                _, grad = self.get_grad_by_loss()
                self.process_grad(grad)
            
            if self.options['outputderiv'] == True:
                pde = self.pde(self.xr,self.model)
                self.process_pde(pde)

            # convert losses to list
            losses = [v.numpy() for _,v in self.current_loss.items()]
            info = [self.iter] + losses

            if self.weighting.method !='constant':
                alphas = [v for _,v in self.weighting.alphas.items()]
                info +=alphas

            if self.model.param is not None:
                info.extend(np.array(list(self.model.param.values())))
            
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
            if self.geomodel is not None:
                save_path = self.managergeo.save()
            
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
            rests.append(t2n(restxr['residual']))
    
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
        savedat['resxr'] = t2n(resxr['residual'])

        if self.geomodel is not None:
            P = self.geomodel(self.xr[:,1:])
            savedat['ppred'] = t2n(P)
        
        
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

        
