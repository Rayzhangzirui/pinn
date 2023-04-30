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

from nn import PINN

# make prediction and save
from scipy.io import savemat

from config import *
from util import *

from weight import Weighting
from rbf_for_tf2.rbflayer import RBFLayer
from tflbfgs import LossAndFlatGradient
import warnings


tf.keras.backend.set_floatx(DTYPE)

glob_trainable_variables = []
glob_grad_by_loss = {} # gradients
glob_grad_stat = {} # statistics of gradients


# debug, not compatible with tf.gradients
# tf.config.run_functions_eagerly(True) 
# RuntimeError: tf.gradients is not supported when eager execution is enabled. Use tf.GradientTape instead.

# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class Logger(object):
    def __init__(self,fname):
        
        self.terminal = sys.stdout
        self.log = open(fname, "w")
        
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class EarlyStopping:
    def __init__(self, monitor:list = ['total'], patience=100, min_delta = 1e-6):
        self.monitor = monitor
        self.patience = patience
        self.best_losses = {loss: float('inf') for loss in self.monitor}
        self.wait = {loss: 0 for loss in self.monitor}
        self.reason = ''  # reason for stopping
        self.min_delta = min_delta # minimum improvement to be considered as improvement
        
    
    def reset(self):
        self.best_losses = {loss: float('inf') for loss in self.monitor}
        self.wait = {loss: 0 for loss in self.monitor}
        
    def check_stop(self, loss_dict:dict, param_dict:dict):
        # Determine the largest improvement across all monitored losses
        stop = False

        # check if rD is too small
        if param_dict['rD']< 0.05:
            stop = True
            self.reason = f'early stop: rD too small'
            return stop

        for loss in self.monitor:
            cur_loss = loss_dict[loss]
            best_cur_loss = self.best_losses[loss]
            # Check if the largest improvement exceeds the
            if cur_loss < best_cur_loss - self.min_delta:
                self.best_losses[loss] = cur_loss
                self.wait[loss] = 0
            else:
                self.wait[loss] += 1
                if self.wait[loss] >= self.patience:
                    stop = True
                    self.reason = f'early stop after {self.patience} due to {loss}'
                    break

        return stop

class PINNSolver():
    def __init__(self, model, pde, 
                losses,
                dataset,
                manager,
                geomodel=None, 
                wr = None,
                options=None):
        self.model = model
        self.geomodel = geomodel
        self.pde = pde
        self.dataset = dataset
        self.manager = manager
        
        self.losses = losses
        

        self.options = options
        
        self.info = {} #empty dictionary to store information

         # weight of residual
        if wr is None:
            self.wr = tf.ones([tf.shape(self.dataset.xr)[0],1], dtype = DTYPE)
        else:
            self.wr = wr

        # Initialize history of losses and global iteration counter
        self.hist = []
        self.iter = 0
        self.header = ''
        self.paramhist = [] # history of trainable model params
        self.current_optimizer = None # current optimizer
        self.gradlog = None #log for gradient

        

        self.earlystop = EarlyStopping(**self.options['earlystop_opts'])
        # set up log
        os.makedirs(options['model_dir'], exist_ok=True)
        
        logfile = os.path.join(options['model_dir'],'solver.log')

        # reset stdout https://stackoverflow.com/questions/14245227/python-reset-stdout-to-normal-after-previously-redirecting-it-to-a-file
        sys.stdout = sys.__stdout__ 
        if self.options.get('file_log'):
            # if log to file
            if os.path.exists(logfile): 
                if os.path.isdir(os.path.join(options['model_dir'],'ckpt')):
                    print(f'{logfile} exist, and ckpt exist, exit')
                    sys.exit()
                else:
                    print(f'{logfile} exist, but ckpt do not exist, continue')
                    
            
            sys.stdout = Logger(logfile)
        else:
            print('skip file log')
        
        
        # may set some layer trainable = False
        self.model.set_trainable_layer(self.options.get('trainnnweight'))

        self.model.summary()

        # collect all trainable variables
        global glob_trainable_variables
        glob_trainable_variables+=  self.model.trainable_variables
        if self.geomodel is not None:
            glob_trainable_variables +=  self.geomodel.trainable_variables

    
    @tf.function
    def get_grad(self):
        """ get loss, residual, gradient
        called by both solve_with_TFoptimizer and solve_with_ScipyOptimizer, 
        !do not use tf.function for dynamic weighting, the weight seems not in the graph
        !if use tf.function, the weight will be fixed to the initial value
        """
        self.losses.trainmode()
        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
            # This tape is for derivatives with
            # respect to trainable variables
            # tape.watch(self.model.trainable_variables)
            tape.watch(glob_trainable_variables)
            loss = self.losses.getloss()
        
        g = tape.gradient(loss['total'], glob_trainable_variables)
        grad_stat = {}
        grad_by_loss = {}
        if self.losses.weighting.method == "grad":
            # compute gradient for each loss
            for loss_name in loss.keys():
                grad_by_loss[loss_name] = tape.gradient(loss[loss_name], glob_trainable_variables)
                flattened_tensors = [tf.reshape(tensor, [-1]) for tensor in grad_by_loss[loss_name]]
                concatenated_tensor = tf.concat(flattened_tensors, axis=0)

                stat = {}
                stat['mean'] =  tf.reduce_mean(tf.abs(concatenated_tensor))
                stat['max'] = tf.reduce_max(tf.abs(concatenated_tensor))
                grad_stat[loss_name] = stat

        del tape
        return loss, g, grad_by_loss, grad_stat
    
     
    
    def solve_with_TFoptimizer(self, optimizer, N=10000):
        """This method performs a gradient descent type optimization."""

        @tf.function
        def train_step():
            loss, grad, grad_by_loss, grad_stat = self.get_grad()
            # Perform gradient descent step
            optimizer.apply_gradients(zip(grad, glob_trainable_variables))
            return loss, grad, grad_by_loss, grad_stat
        
        global glob_grad_by_loss
        global glob_grad_stat
        
        self.current_optimizer = self.options['optimizer']
        start = time()
        

        for i in range(N):
            loss, grad, glob_grad_by_loss, glob_grad_stat = train_step()
            self.current_loss = loss # before applying gradient
            
            try:
                self.callback()
            except Exception as e:
                print(e)
                break

        end = time()

        self.info['tfadamiter'] = self.iter
        self.info['tfadamtime'] = (end-start)
        self.info['tfadamloss'] = self.current_loss
        print('adam It:{:05d}, loss {:10.4e}, time {}'.format(i, loss['total'].numpy(), end-start))
        
        self.losses.weighting.active = False
        print('turn off dynamic weighting after Adam')

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

            global glob_grad_by_loss
            global glob_grad_stat
            # Determine value of \phi and gradient w.r.t. \theta at w
            loss_dict, grad, glob_grad_by_loss, glob_grad_stat = self.get_grad()
            
            # Store current loss for callback function            
            loss = loss_dict['total'].numpy().astype(np.float64)
            self.current_loss = loss_dict # before applying gradient in lbfgs
            
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
        
        # the output of callback is not used in lbfgsb
        # use exception
        # https://github.com/scipy/scipy/blob/v1.10.1/scipy/optimize/_lbfgsb_py.py
        try:
            results = scipy.optimize.minimize(fun=get_loss_and_grad,
                                       x0=x0,
                                       jac=True,
                                       method=method,
                                       callback=self.callback,
                                       **kwargs)
            print('lbfgs(scipy) It:{:05d}, loss {:10.4e}, {}.'.format(results.nit, results.fun, results.message ))
        except Exception as e:
            print(e)
            print('lbfgs(scipy) stop early')
            pass
        
        if self.info.get('tfadamiter') is None:
            self.info['scipylbfgsiter'] = self.iter
        else:
            self.info['scipylbfgsiter'] = self.iter - self.info.get('tfadamiter')
        end = time()
        
        self.info['scipylbfgstime'] = (end-start)
        self.info['scipylbfgslosses'] = self.current_loss
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult
        print(f'lbfgs(scipy) time {end-start}')
        self.callback_train_end()


    # not used
    @timer
    def solve_with_tfbfgs(self,**kwargs):
        
        def bfgs_loss():
            self.losses.trainmode()
            dloss = self.losses.getloss()
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
        pdedict['xr'] = self.dataset.xr
        
        dirpath = os.path.join(self.options['model_dir'], 'pde')
        if self.iter == 1:
            os.makedirs(dirpath, exist_ok=True)

        predfile = os.path.join(dirpath,f'pde_{self.iter}.mat')
        savemat(predfile,pdedict)
        return None


    def callback(self,xk=None):
        """ called after one step of iteration in bfgs and adam, 
        after application of gradient.
        scipy.optimize.minimize require first arg to be parameters 
        if callback return true, ealy stop
        """
        self.iter+=1
        # update weights in model
        self.losses.weighting.update_weights(self.current_loss, glob_grad_stat)
        
        # compute all patient data loss 
        self.losses.testmode()
        test_losses = self.losses.getloss()
        test_losses = {key + 'test': value for key, value in test_losses.items()}

        test_monitor = tf.Variable(0., dtype=DTYPE)
        # `variable += value` with `tf.Variable`s is not supported. Use `variable.assign_add(value)` to modify the variable, or `out = variable + value` if you need to get a new output Tensor.
        for x in self.losses.data_test_loss:
            test_monitor = test_monitor + test_losses[x + 'test']
            
        test_losses['pdattest'] = test_monitor
        
        all_losses = self.current_loss | test_losses
        earlystop = self.earlystop.check_stop(all_losses, self.model.param)
        if earlystop:
            print(self.earlystop.reason)
            raise Exception
            # https://github.com/scipy/scipy/blob/v1.10.1/scipy/optimize/_lbfgsb_py.py#L48-L207
            # lbfgs returning true in callback does not stop the mnimizer
        
            

        # in the first iteration, create header
        # this is loss after gradient step
        if self.iter == 1:
            trainable_params = []
            # create header
            str_losses = ','.join('{:<12}'.format(k) for k in self.current_loss)
            header = '{:<6}, {:<12}'.format('it',str_losses)
            if self.losses.weighting.method != 'constant':
                str_weight = ','.join('W{:<12}'.format(k) for k in self.losses.weighting.weight_keys) 
                header += ', {}'.format(str_weight)
            if self.model.param is not None:
                # if not none, add to header
                for pname,ptensor in self.model.param.items():
                    if ptensor.trainable == True:
                        header+= ", {:<12}".format(f'{pname}')
            if self.losses.hastest == True:
                for lname in self.losses.all_test_losses:
                    header+= ",{:<12}".format(lname+'test')
                header+= ",{:<12}".format('pdattest') # patient data loss
            self.header = header
            print(header)
        
        # write to file if iter==1 or print_res_every
        yeswrite = (self.iter % self.options['print_res_every'] == 0) or (self.iter==1)
        if yeswrite:

            if self.options['gradnorm'] == True:
                sys.exit('not updated when introducing losses class')
                _, grad = self.get_grad_by_loss()
                self.process_grad(grad)
            
            if self.options['outputderiv'] == True:
                sys.exit('not updated when introducing losses class')
                pde = self.pde(self.dataset.xr,self.model)
                self.process_pde(pde)

            # convert losses to list
            losses = [v.numpy() for _,v in self.current_loss.items()]
            info = [self.iter] + losses

            if self.losses.weighting.method !='constant':
                alphas = [v.numpy() for _,v in self.losses.weighting.alphas.items()]
                info += alphas

            if self.model.param is not None:
                for pname,ptensor in self.model.param.items():
                    if ptensor.trainable == True:
                        info.append(ptensor.numpy())
            
            # if provide test data, output test mse
            if self.losses.hastest == True:
                for lname in self.losses.all_test_losses:
                    info.append(test_losses[lname + 'test'].numpy())
                info.append(test_monitor.numpy())
            
            info_str = ','.join('{:<12.4e}'.format(k) for k in info[1:])
            print('{:05d}, {}'.format(info[0], info_str))  
            self.hist.append(info)
            # breakpoint()
        
        


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
        self.earlystop.reset()
        if self.options.get('saveckpt'):
            save_path = self.manager.save()
            print("Saved checkpoint for {} step {} {}".format(int(self.iter),self.current_optimizer, save_path))
            if self.geomodel is not None:
                save_path = self.managergeo.save()
            
        else:
            print("checkpoint not saved")

        self.predtx(self.current_optimizer, 1.0)
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
        xr = np.copy(self.dataset.xr)
        for t in ts:
            xr[:,0] = t

            upredtxr = self.model(xr)
            if self.dataset.xdim == 2:
                restxr = self.pde(xr, self.model, self.dataset.phiq, self.dataset.Pq, self.dataset.DxPphi, self.dataset.DyPphi)
            else:
                restxr = self.pde(xr, self.model, self.dataset.phiq, self.dataset.Pq, self.dataset.DxPphi, self.dataset.DyPphi, self.dataset.DzPphi)
            
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

        self.losses.savemode()
        self.losses.getupredxr()
        self.losses.getupredxdat()
        
        savedat['xr'] = t2n(self.dataset.xr)
        savedat['upredxr'] = t2n(self.losses.upredxr)

        if self.dataset.xdat is not None:
            upredxdat = self.model(self.dataset.xdat)
            savedat['xdat'] = t2n(self.dataset.xdat)
            savedat['upredxdat'] = t2n(self.losses.upredxdat)

        # may not have current_loss if reload
        # savedat['lossname'] = [k for k,v in self.current_loss.items()]
        # savedat['lossval'] = [v.numpy() for k,v in self.current_loss.items()]
        self.losses.getpdeterm()
        savedat.update(t2n(self.losses.pdeterm))

        if self.geomodel is not None:
            P = self.geomodel(self.dataset.xr[:,1:])
            savedat['ppred'] = t2n(P)

        for key in self.model.param:
            savedat[key] = self.model.param[key].numpy()

        predfile = os.path.join(self.options['model_dir'],f'upred_{suffix}.mat')
        
        print(f'save upred to {predfile}')
        savemat(predfile,savedat)

    def reweight(self, topk):
        sys.exit('outdated')
        res = np.abs(self.pde(self.dataset.xr, self.model).numpy().flatten()) # compute residual
        wres = self.wr
        # get topk idx
        # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        ind = np.argpartition(res, -topk)[-topk:]
        wres[ind]+=1
        self.wr = np.reshape(wres,(-1,1))
    
    def resample(self, topk, xxr):
        sys.exit('outdated')
        ''' compute residual on xxr, add topk points to collocation pts
        '''
        res = np.abs(self.pde(xxr, self.model).numpy().flatten())
        ind = np.argpartition(res, -topk)[-topk:]
        xxr = xxr.numpy()[ind,:]
        newx = tf.convert_to_tensor(xxr)
        self.dataset.xr = tf.concat([self.dataset.xr, newx],axis=0)
        self.wr = np.ones([tf.shape(self.dataset.xr)[0],1])
        return xxr

        
