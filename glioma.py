#!/usr/bin/env python
# solve fisher-komogorov equation in 2d

from re import S
import sys
sys.path.insert(1, '/home/ziruz16/pinn')

import json

from config import *
from pinn import *

import tensorflow_probability as tfp


class DataSet:
    def __init__(self, opts) -> None:
        self.opts = opts
        
        
        inv_dat_file = opts['inv_dat_file']

        assert os.path.exists(inv_dat_file), f'{inv_dat_file} not exist'
        
        _ , ext = os.path.splitext(inv_dat_file)
        
        assert ext == '.mat', 'not reading mat file'
        

        matdat = loadmat(inv_dat_file)

        for key, value in matdat.items():
            if isinstance(value,np.ndarray) and value.dtype.kind == 'f':
                matdat[key] = value.astype(DTYPE)

                
        ntest = opts.get("n_test_pts", matdat.get('xtest').shape[0])  
        ndat =  opts.get("n_dat_pts",  matdat.get('xdat').shape[0]) 
        nres =  opts.get("n_res_pts",  matdat.get('xr').shape[0])
        
        # testing data
        self.xtest = matdat.get('xtest')[0:ntest,:]
        self.utest = matdat.get('utest')[0:ntest,:] 
        self.phitest = matdat.get('phitest')[0:ntest,:]


        self.xbc = matdat.get('xbc')
        self.ubc = matdat.get('ubc') 
        self.phibc = matdat.get('phibc')

        # data loss
    
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
            self.plfdat = matdat.get('plfdat')[0:ndat,:] 
        
        if opts.get('addnoise') is not None:
            print('add noise to udat')
            self.udat = self.udat + np.random.normal(scale = opts.get('addnoise'), size = self.udat.shape)

        
        # non dimensional parameters
        self.T = matdat['T'].item()
        self.L = matdat['L'].item()
        self.DW = matdat['DW'].item() #characteristic DW
        self.RHO = matdat['RHO'].item() #characteristic RHO
        self.rDe = matdat['rDe'].item() # exact ratio
        self.rRHOe = matdat['rRHOe'].item()
        self.opts['scale'] = {'T':self.T, 'L':self.L, 'DW':self.DW, 'RHO':self.RHO}

        # residual pts
        self.xr =  matdat.get('xr')[0:nres,:]
        self.P =  matdat.get('Pq')[0:nres,:]
        self.phi =  matdat.get('phiq')[0:nres,:]
        self.DxPphi =  matdat.get('DxPphi')[0:nres,:]
        self.DyPphi =  matdat.get('DyPphi')[0:nres,:]
        self.DzPphi =  matdat.get('DzPphi')[0:nres,:]

        # characteristic diffusion ceofficient at each point
        self.dim = self.xr.shape[1]
        self.xdim = self.xr.shape[1]-1


class Gmodel:
    def __init__(self, opts) -> None:

        self.dataset = DataSet(opts)
        self.dim = self.dataset.dim
        self.xdim = self.dataset.xdim
        self.opts = opts

        if opts.get('optimizer') == 'adamax':
            self.optim = tf.keras.optimizers.Adamax()
        elif opts.get('optimizer') == 'rmsprop':
            self.optim = tf.keras.optimizers.RMSprop()
        else:
            self.opts['optimizer'] = 'adam'
            self.optim = tf.keras.optimizers.Adam()

        if opts.get('exactfwd') == True:
            print('use exat parameter from dataset')
            opts['D0'] = self.dataset.rDe
            opts['RHO0'] = self.dataset.rRHOe

        param = {'rD':tf.Variable( opts['D0'], trainable=opts.get('trainD'), dtype = DTYPE),
        'rRHO': tf.Variable(opts['RHO0'], trainable=opts.get('trainRHO'),dtype = DTYPE)}

        self.info = {}

        def ic(x):
            L = self.dataset.L
            r2 = tf.reduce_sum(tf.square(x[:, 1:self.dim]*L),1,keepdims=True) # this is in pixel scale, unit mm, 
            return 0.1*tf.exp(-0.1*r2)

        
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
            if self.opts.get('exactres') == True:
                print('use exact residual')
                @tf.function
                def pde(x_r, f):
                    t = x_r[:,0:1]
                    x = x_r[:,1:2]
                    y = x_r[:,2:3]
                    xr = tf.concat([t,x,y], axis=1)
                    
                    r = tf.sqrt((x*self.dataset.L)**2+(y*self.dataset.L)**2)
                    phi = 0.5 + 0.5*tf.tanh((50.0 - r)/1.0)
                    P = 0.9*( 0.5 + 0.5*tf.tanh((20.0 - r)/1.0)) + 0.1

                    u =  f(xr)
                    
                    u_t = tf.gradients(u, t)[0]

                    u_x = tf.gradients(u, x)[0]
                    u_y = tf.gradients(u, y)[0]
                    
                    u_xx = tf.gradients( u_x, x)[0]
                    u_yy = tf.gradients( u_y, y)[0]
                    
                    DxPphi = tf.gradients( P * phi, x)[0]
                    DyPphi = tf.gradients( P * phi, y)[0]

                    diffusion =  f.param['rD'] * self.dataset.DW *( P * phi * (u_xx + u_yy) + DxPphi * u_x + DyPphi * u_y)
                    
                    prolif = f.param['rRHO'] * self.dataset.RHO * phi * u * (1-u)

                    res = phi * u_t - ( diffusion + prolif)
                    return res
            else:
                @tf.function
                def pde(x_r, f):
                    t = x_r[:,0:1]
                    x = x_r[:,1:2]
                    y = x_r[:,2:3]
                    xr = tf.concat([t,x,y], axis=1)
                    
                    u =  f(xr)
                    
                    u_t = tf.gradients(u, t)[0]

                    u_x = tf.gradients(u, x)[0]
                    u_y = tf.gradients(u, y)[0]
                    
                    u_xx = tf.gradients(u_x, x)[0]
                    u_yy = tf.gradients(u_y, y)[0]

                    prolif = f.param['rRHO'] * self.dataset.RHO * self.dataset.phi * u * (1-u)

                    diffusion = f.param['rD'] * self.dataset.DW * (self.dataset.P *self.dataset.phi * (u_xx + u_yy) + self.dataset.L* self.dataset.DxPphi * u_x + self.dataset.L* self.dataset.DyPphi * u_y)
                    res = self.dataset.phi * u_t - ( diffusion +  prolif)
                    return res

            
            
        else:
            @tf.function
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
        # def fdatloss(nn, xdat):
        #     upred = nn(xdat)
        #     loss = tf.math.reduce_mean(tf.math.square((self.dataset.udat - upred)*self.dataset.phidat))
        #     return loss
        def bcloss(nn):
            upredbc = nn(self.dataset.xbc)
            loss = tf.math.reduce_mean(tf.math.square((self.dataset.ubc - upredbc)*self.dataset.phibc))
            return loss

        def fdatloss(nn, xdat):

            bc_loss = bcloss(nn)
            
            upred = nn(xdat)
            # neg_loss = tf.reduce_mean(tf.nn.relu(-upred)**2)
            prolif = 4 * upred * (1-upred)
            cor_loss = - tfp.stats.correlation(prolif*self.dataset.phidat, self.dataset.plfdat*self.dataset.phidat)
            
            loss =  cor_loss + bc_loss
            
            return tf.squeeze(loss)
        
        

        
        def ftestloss(nn, xtest):
            upred = nn(xtest)
            loss = tf.math.reduce_mean(tf.math.square((self.dataset.utest - upred)*self.dataset.phitest))
            return tf.squeeze(loss)
        

        self.model = PINN(param=param,
                input_dim=self.dim,
                activation = opts['activation'],
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
        if self.opts["num_init_train"] > 0:
            self.solver.solve_with_TFoptimizer(self.optim, N=self.opts["num_init_train"],patience = self.opts["patience"])

        if self.opts['lbfgs_opts'] is not None:
            results = self.solver.solve_with_ScipyOptimizer(method='L-BFGS-B', options=self.opts['lbfgs_opts'])
        
        self.saveopts()
        
    def saveopts(self):
        # save all options 
        z = self.opts | self.solver.info
        fpath = os.path.join(self.opts['model_dir'],'options.json')
        json.dump( z, open( fpath, 'w' ) )