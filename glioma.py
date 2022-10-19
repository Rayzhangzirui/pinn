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
        

        ntest = opts.get("n_test_pts", matdat.get('xtest').shape[0])  
        ndat =  opts.get("n_dat_pts",  matdat.get('xdat').shape[0]) 
        nres =  opts.get("n_res_pts",  matdat.get('xr').shape[0])
        
        # testing data
        self.xtest = matdat.get('xtest')[0:ntest,:]
        self.utest = matdat.get('utest')[0:ntest,:] 
        self.phitest = matdat.get('phitest')[0:ntest,:]

        # data loss
        if opts['w_dat'] == 0:
            self.xdat = None
            self.udat = None # final time time
        else:
            if opts.get('testasdat') == True:
                print('use test data for data loss')
                # use test data as data loss, full time
                self.xdat = self.xtest
                self.udat = self.utest
                self.phidat = self.phitest
            else:
                # single time inference
                self.xdat = matdat.get('xdat')[0:ndat,:]
                self.udat = matdat.get('udat')[0:ndat,:] 
                self.phidat = matdat.get('phidat')[0:ndat,:]
            
        # residual pts
        self.xr =  matdat.get('xr')[0:nres,:]
        self.phi =  matdat.get('phiq')[0:nres,:]
        self.pwm =  matdat.get('Pwmq')[0:nres,:]
        self.pgm =  matdat.get('Pgmq')[0:nres,:]

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
        train_rD = True if opts.get('train_rD') == True else False
        train_rRHO = True if opts.get('train_rRHO') == True else False

        param = {'rD':tf.Variable( opts['D0'], trainable=train_rD), 'rRHO': tf.Variable(opts['RHO0'], trainable=train_rRHO)}

        self.info = {}

        def ic(x):
            r2 = tf.reduce_sum(tf.square(x[:, 1:self.dim]),1,keepdims=True)
            return 0.1*tf.exp(-0.1*r2*(self.dataset.L**2))
        
        if opts.get('ictransofrm') == False:
            # without output transform, ic as data loss
            print('no nn ic transformation')
            def ot(x,u):
                return u
        else:
            print('apply nn ic transformation')
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

        # loss function of data, difference of phi * u
        def fdatloss(nn, xdat):
            upred = nn(xdat)
            loss = tf.math.reduce_mean(tf.math.square((self.dataset.udat - upred)*self.dataset.phidat))
            return loss
        
        def ftestloss(nn, xtest):
            upred = nn(xtest)
            loss = tf.math.reduce_mean(tf.math.square((self.dataset.utest - upred)*self.dataset.phitest))
            return loss
        

        self.model = PINN(param=param,
                input_dim=self.dim,
                num_hidden_layers=opts["num_hidden_layer"], 
                num_neurons_per_layer=opts["num_hidden_unit"],
                output_transform=ot)


        # Initilize PINN solver
        self.solver = PINNSolver(self.model, pde, 
                                fdatloss,
                                ftestloss,
                                xr = self.dataset.xr,
                                xdat = self.dataset.xdat,
                                udat = self.dataset.udat,
                                xtest= self.dataset.xtest,
                                utest= self.dataset.utest,
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