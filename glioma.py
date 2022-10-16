#!/usr/bin/env python
# solve fisher-komogorov equation in 2d

from re import S
import sys
sys.path.insert(1, '/home/ziruz16/pinn')

import json

from config import *
from pinn import *

class DataSet:
    def __init__(self, opts) -> None:
        self.opts = opts
        
        
        inv_dat_file = opts['inv_dat_file']

        assert os.path.exists(inv_dat_file), f'{inv_dat_file} not exist'
        
        _ , ext = os.path.splitext(inv_dat_file)
        
        assert ext == '.mat', 'not reading mat file'
        
        matdat = loadmat(inv_dat_file)
        if opts["n_res_pts"] is not None:
            n = opts["n_res_pts"]
        else:
            n = matdat.get('xtest').shape[0]
        
        self.xtest = matdat.get('xtest')[0:n,:]
        self.phiutest = matdat.get('phiutest')[0:n,:] #all time

        if opts['w_dat'] == 0:
            self.xdat = None
            self.phiudat = None # final time time
        else:
            if opts.get('testasdat') == True:
                # use test data as data loss, full time
                self.xdat = self.xtest
                self.phiudat = self.phiutest
            else:
                # single time inference
                self.xdat = matdat.get('xdat')[0:n,:]
                self.phiudat = matdat.get('phiudat')[0:n,:] # final time time
            

        self.xr = self.xtest

        self.phi =  matdat.get('phiq')[0:n,:]
        self.pwm =  matdat.get('Pwmq')[0:n,:]
        self.pgm =  matdat.get('Pgmq')[0:n,:]

        # non dimensional parameters
        self.T = matdat['T'].item()
        self.L = matdat['L'].item()
        self.DW = matdat['DW'].item() #characteristic DW
        self.RHO = matdat['RHO'].item() #characteristic RHO
        self.opts['scale'] = {'T':self.T, 'L':self.L, 'DW':self.DW, 'RHO':self.RHO}

        # characteristic diffusion ceofficient at each point
        self.Dphi = (self.pwm * self.DW + self.pgm * (self.DW/10)) * self.phi        
        self.dim = self.xr.shape[1]
        self.xdim = self.xr.shape[1]-1

class Gmodel:
    def __init__(self, opts) -> None:

        self.dataset = DataSet(opts)
        self.dim = self.dataset.dim
        self.xdim = self.dataset.xdim
        self.opts = opts

        self.optim = tf.keras.optimizers.Adam(learning_rate=1e-3)

        INVERSE = opts.get('inverse')
        param = {'rD':tf.Variable( opts['D0'], trainable=INVERSE), 'rRHO': tf.Variable(opts['RHO0'], trainable=INVERSE)}

        self.info = {}

        def ic(x):
            r2 = tf.reduce_sum(tf.square(x[:, 1:self.dim]),1,keepdims=True)
            return 0.1*tf.exp(-0.1*r2*(self.dataset.L**2))
        
        def ot(x,u):
            return u* x[:, 0:1]+ ic(x)
        
        if self.xdim == 2:
            def pde(x_r, f):
                t = x_r[:,0:1]
                x = x_r[:,1:2]
                y = x_r[:,2:3]
                xr = tf.concat([t,x,y], axis=1)
                u =  f(xr)
                
                u_t = tf.gradients(u, t)[0]

                u_x = tf.gradients(u, x)[0]
                u_xx = tf.gradients(self.dataset.Dphi*u_x, x)[0]
                
                u_y = tf.gradients(u, y)[0]
                u_yy = tf.gradients(self.dataset.Dphi*u_y, y)[0]

                res = self.dataset.phi*u_t - (f.param['rD'] * (u_xx + u_yy) + f.param['rRHO'] * self.dataset.RHO * self.dataset.phi * u * (1-u))
                return res
        else:
            def pde(x_r, f):
                 # t,x,y normalized here
                t = x_r[:,0:1]
                x = x_r[:,1:2]
                y = x_r[:,2:3]
                z = x_r[:,3:4]
                xr = tf.concat([t,x,y,z], axis=1)
                u =  f(xr)
                
                u_t = tf.gradients(u, t)[0]

                u_x = tf.gradients(u, x)[0]
                u_xx = tf.gradients(self.dataset.Dphi*u_x, x)[0]
                
                u_y = tf.gradients(u, y)[0]
                u_yy = tf.gradients(self.dataset.Dphi*u_y, y)[0]

                u_z = tf.gradients(u, z)[0]
                u_zz = tf.gradients(self.dataset.Dphi*u_z, z)[0]

                res = self.dataset.phi*u_t - (f.param['rD'] * (u_xx + u_yy + u_zz) + f.param['rRHO'] * self.dataset.RHO * self.dataset.phi * u * (1-u))
                return res



        self.model = PINN(param=param,
                input_dim=self.dim,
                num_hidden_layers=opts["num_hidden_layer"], 
                num_neurons_per_layer=opts["num_hidden_unit"],
                output_transform=ot)

        # Initilize PINN solver
        self.solver = PINNSolver(self.model, pde, 
                                xr = self.dataset.xr,
                                xdat = self.dataset.xdat,
                                udat = self.dataset.phiudat,
                                xtest= self.dataset.xtest,
                                utest= self.dataset.phiutest,
                                options = opts)
    
    def solve(self):
        self.solver.solve_with_TFoptimizer(self.optim, N=self.opts["num_init_train"],patience = self.opts["patience"])

        if self.opts['lbfgs_opts'] is not None:
            results = self.solver.solve_with_ScipyOptimizer(method='L-BFGS-B', options=self.opts['lbfgs_opts'])

        self.saveopts()
        
    def saveopts(self):
        z = self.opts | self.solver.info
        fpath = os.path.join(self.opts['model_dir'],'options.json')
        json.dump( z, open( fpath, 'w' ) )