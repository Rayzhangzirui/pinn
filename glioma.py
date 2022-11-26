#!/usr/bin/env python
# solve fisher-komogorov equation in 2d

from re import S
import sys
sys.path.insert(1, '/home/ziruz16/pinn')

import json

from config import *
from pinn import *
from DataSet import DataSet

import tensorflow_probability as tfp



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

                    prolif = f.param['rRHO'] * self.dataset.RHO * self.dataset.phiq * u * (1-u)

                    diffusion = f.param['rD'] * self.dataset.DW * (self.dataset.Pq *self.dataset.phiq * (u_xx + u_yy) + self.dataset.L* self.dataset.DxPphi * u_x + self.dataset.L* self.dataset.DyPphi * u_y)
                    res = self.dataset.phiq * u_t - ( diffusion +  prolif)
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

                res = self.dataset.phiq*u_t - (f.param['rD'] * (u_xx + u_yy + u_zz) + f.param['rRHO'] * self.dataset.RHO * self.dataset.phiq * u * (1-u))
                return res
        
        @tf.function
        def grad(X, f):
            # t,x,y normalized here
            t = X[:,0:1]
            x = X[:,1:2]
            y = X[:,2:3]
            Xcat = tf.concat([t,x,y], axis=1)
            u =  f(Xcat)
            u_x = tf.gradients(u, x)[0]
            u_y = tf.gradients(u, y)[0]
            return u, u_x, u_y

        # loss function of data, difference of phi * u
        def fdatloss(nn):
            upred = nn(self.dataset.xdat)
            loss = tf.math.reduce_mean(tf.math.square((self.dataset.udat - upred)*self.dataset.phidat))
            return loss

        def profmseloss(nn):
            upred = nn(self.dataset.xdat)
            prolif = 4 * upred * (1-upred)

            loss = tf.math.reduce_mean(tf.math.square((self.dataset.plfdat - prolif)*self.dataset.phidat))
            return loss

        def bcloss(nn):
            upredbc = nn(self.dataset.xbc)
            loss = tf.math.reduce_mean(tf.math.square((self.dataset.ubc - upredbc)*self.dataset.phibc))
            return loss

        def negloss(nn):
            # penalize negative of u
            upred = nn(self.dataset.xdat)
            neg_loss = tf.reduce_mean(tf.nn.relu(-upred)**2)
            return neg_loss

        def fcorloss(nn):
            # correlation of proliferation 4u(1-u)
            upred = nn(self.dataset.xdat)
            prolif = 4 * upred * (1-upred)
            loss =  - tfp.stats.correlation(prolif*self.dataset.phidat, self.dataset.plfdat*self.dataset.phidat)
            loss = tf.squeeze(loss)
            return loss
        
        def fgradcorloss(nn):
            # correlation of gradient, assume 2D
            u, ux, uy = grad(self.dataset.xdat, nn)
            dxprolif = 4 * (ux-2*u*ux)
            dyprolif = 4 * (uy-2*u*uy)
            loss =  - tfp.stats.correlation(dxprolif, self.dataset.dxplfdat) - tfp.stats.correlation(dyprolif, self.dataset.dyplfdat) 
            loss = tf.squeeze(loss)
            return loss
        
        def ftestloss(nn):
            upred = nn(self.dataset.xtest)
            loss = tf.math.reduce_mean(tf.math.square((self.dataset.utest - upred)*self.dataset.phitest))
            return loss
        

        self.model = PINN(param=param,
                input_dim=self.dim,
                activation = opts['activation'],
                num_hidden_layers=opts["num_hidden_layer"], 
                num_neurons_per_layer=opts["num_hidden_unit"],
                output_transform=ot)

        flosses = {'gradcor': fgradcorloss ,'bc':bcloss, 'cor':fcorloss}

        ftest = {'test':ftestloss} 

        # Initilize PINN solver
        self.solver = PINNSolver(self.model, pde, 
                                flosses,
                                ftest,
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