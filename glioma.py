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

        self.opts = opts

        self.dataset = DataSet(opts['inv_dat_file'])
        if self.opts.get('N') is not None:
            self.dataset.downsample(self.opts.get('N'))
        if opts.get('useupred') is not None:
            # use upred at xdat from other training
            tmpdataset = DataSet(opts['useupred'])
            self.dataset.xdat = np.copy(tmpdataset.xdat)
            self.dataset.udat = np.copy(tmpdataset.upredxdat)
        

        self.dim = self.dataset.dim
        self.xdim = self.dataset.xdim
        

        if opts.get('optimizer') == 'adamax':
            self.optim = tf.keras.optimizers.Adamax()
        elif opts.get('optimizer') == 'rmsprop':
            self.optim = tf.keras.optimizers.RMSprop()
        else:
            self.opts['optimizer'] = 'adam'
            self.optim = tf.keras.optimizers.Adam(learning_rate=opts['lr'])

        if opts.get('exactfwd') == True:
            print('use exat parameter from dataset')
            opts['D0'] = self.dataset.rDe
            opts['RHO0'] = self.dataset.rRHOe
        
        



        self.param = {'rD':tf.Variable( opts['D0'], trainable=opts.get('trainD'), dtype = DTYPE, name="rD"),
        'rRHO': tf.Variable(opts['RHO0'], trainable=opts.get('trainRHO'),dtype = DTYPE,name="rRHO"),
        'M': tf.Variable(opts['M0'], trainable=opts.get('trainM'),dtype = DTYPE,name="M"),
        'm': tf.Variable(opts['m0'], trainable=opts.get('trainm'),dtype = DTYPE,name="m"),
        'x0': tf.Variable(opts['x0'], trainable=opts.get('trainx0'),dtype = DTYPE,name="x0"),
        'y0': tf.Variable(opts['y0'], trainable=opts.get('trainx0'),dtype = DTYPE,name="y0"),
        }

        self.ix = [[self.param['x0'],self.param['y0']]]

        # for computing residual at initial time
        self.dataset.xr0 = np.copy(self.dataset.xr)
        self.dataset.xr0[:,0] = 0.0

        self.info = {}

        def ic(x):
            L = self.dataset.L
            r2 = tf.reduce_sum(tf.square((x[:, 1:self.dim]-self.ix)*L), 1, keepdims=True) # this is in pixel scale, unit mm, 
            return 0.1*tf.exp(-0.1*r2)*self.param['M']

        
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
                exit();
                # NEED update residual
                # print('use exact residual')
                # @tf.function
                # def pde(x_r, f):
                #     t = x_r[:,0:1]
                #     x = x_r[:,1:2]
                #     y = x_r[:,2:3]
                #     xr = tf.concat([t,x,y], axis=1)
                    
                #     r = tf.sqrt((x*self.dataset.L)**2+(y*self.dataset.L)**2)
                #     phi = 0.5 + 0.5*tf.tanh((50.0 - r)/1.0)
                #     P = 0.9*( 0.5 + 0.5*tf.tanh((20.0 - r)/1.0)) + 0.1

                #     u =  f(xr)
                    
                #     u_t = tf.gradients(u, t)[0]

                #     u_x = tf.gradients(u, x)[0]
                #     u_y = tf.gradients(u, y)[0]
                    
                #     u_xx = tf.gradients( u_x, x)[0]
                #     u_yy = tf.gradients( u_y, y)[0]
                    
                #     DxPphi = tf.gradients( P * phi, x)[0]
                #     DyPphi = tf.gradients( P * phi, y)[0]

                #     diffusion =  self.param['rD'] * self.dataset.DW *( P * phi * (u_xx + u_yy) + DxPphi * u_x + DyPphi * u_y)
                    
                #     prolif = self.param['rRHO'] * self.dataset.RHO * phi * u * (1-u)

                #     res = phi * u_t - ( diffusion + prolif)
                #     return res
            else:
                @tf.function
                def pde(x_r, nn):
                    t = x_r[:,0:1]
                    x = x_r[:,1:2]
                    y = x_r[:,2:3]
                    xr = tf.concat([t,x,y], axis=1)
                    
                    u =  nn(xr)
                    
                    u_t = tf.gradients(u, t)[0]

                    u_x = tf.gradients(u, x)[0]
                    u_y = tf.gradients(u, y)[0]
                    
                    u_xx = tf.gradients(u_x, x)[0]
                    u_yy = tf.gradients(u_y, y)[0]

                    prolif = self.param['rRHO'] * self.dataset.RHO * self.dataset.phiq * u * ( 1 - u/self.param['M'])

                    diffusion = self.param['rD'] * self.dataset.DW * (self.dataset.Pq *self.dataset.phiq * (u_xx + u_yy) + self.dataset.L* self.dataset.DxPphi * u_x + self.dataset.L* self.dataset.DyPphi * u_y)
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
                u_xx = tf.gradients(u_x, x)[0]
                
                u_y = tf.gradients(u, y)[0]
                u_yy = tf.gradients(u_y, y)[0]
                
                u_z = tf.gradients(u, z)[0]
                u_zz = tf.gradients(u_z, z)[0]

                prolif = self.param['rRHO'] * self.dataset.RHO * self.dataset.phiq * u * ( 1 - u/self.param['M'])
                diffusion = self.param['rD'] * self.dataset.DW * (self.dataset.Pq *self.dataset.phiq * (u_xx + u_yy + u_zz) 
                + self.dataset.L* (self.dataset.DxPphi * u_x + self.dataset.DyPphi * u_y + self.dataset.DzPphi * u_z))

                res = self.dataset.phiq * u_t - ( diffusion +  prolif)
                return res
        
        @tf.function
        def grad(X, nn):
            # t,x,y normalized here
            t = X[:,0:1]
            x = X[:,1:2]
            y = X[:,2:3]
            Xcat = tf.concat([t,x,y], axis=1)
            u =  nn(Xcat)
            u_x = tf.gradients(u, x)[0]
            u_y = tf.gradients(u, y)[0]
            return u, u_x, u_y
        
        def binarize(x, th):
            # https://stackoverflow.com/questions/37743574/hard-limiting-threshold-activation-function-in-tensorflow
            # return 1.0 if x > th
            # this might give None gradient!
            cond = tf.greater(x, th)
            out = tf.where(cond, 1.0, 0.0) 
            return out
        
        def sigmoidbinarize(x, th):
            # smooth heaviside function using a sigmoid
            return tf.math.sigmoid(50*(x-th))

        def smoothheaviside(x, th):
            # smooth heaviside function
            # tfp.math.smootherstep S(x), y goes from 0 to 1 as x goes from 0 1
            # F(x) = S((x+1)/2) = S(x/2+1/2), -1 to 1
            # G(x) = S(Kx/2+1/2), -1/K to 1/K
            K = 10.0
            return tfp.math.smootherstep(K * (x-th)/2.0 + 1.0/2.0)
             

        
        def dice(T, P):
            # x is prediction (pos, neg), y is label,
            TP = tf.reduce_sum(T*P)
            FP = tf.reduce_sum((1-T)*P)
            FN = tf.reduce_sum(T*(1-P))
            return 2 * TP / (2*TP + FP + FN)

        # maximize dice, minimize 1-dice
        def fdice1loss(nn): 
            upred = nn(self.dataset.xdat)
            pu1 = sigmoidbinarize(upred, self.dataset.seg[0,0])
            d1 = dice(self.dataset.u1, pu1)
            return 1.0-d1

        def fdice2loss(nn): 
            upred = nn(self.dataset.xdat)
            pu2 = sigmoidbinarize(upred, self.dataset.seg[0,1])
            d2 = dice(self.dataset.u2, pu2)
            return 1.0-d2
        
        # segmentation mse loss, mse of threholded u and data (patient geometry)
        def fseg1loss(nn): 
            upred = nn(self.dataset.xdat)
            pu1 = sigmoidbinarize(upred, self.dataset.seg[0,0])
            # pu1 = smoothheaviside(upred, self.dataset.seg[0,0])
            return tf.reduce_mean((self.dataset.phidat*(pu1-self.dataset.u1))**2)

        def fseg2loss(nn): 
            upred = nn(self.dataset.xdat)
            pu2 = sigmoidbinarize(upred, self.dataset.seg[0,1])
            # pu2 = smoothheaviside(upred, self.dataset.seg[0,1])
            return tf.reduce_mean((self.dataset.phidat*(pu2-self.dataset.u2))**2)

        def neg_mse(x):
            return tf.reduce_mean(tf.where(x > 0, 0.0, 0.5 * x**2))

        def fseglower1loss(nn): 
            # if upred>u1, no loss, otherwise, mse loss
            upred = nn(self.dataset.xdat)
            diff = self.dataset.phidat*(upred - self.dataset.u1)
            return  neg_mse(diff)

        def fseglower2loss(nn): 
            # if upred>u1, no loss, otherwise, mse loss
            upred = nn(self.dataset.xdat)
            diff = self.dataset.phidat*(upred - self.dataset.u2)
            return  neg_mse(diff)

        def fadcmseloss(nn):
            # error of adc prediction,
            # this adc is ratio w.r.t characteristic adc
            upred = nn(self.dataset.xdat)
            predadc = (1.0 - self.param['m']* upred)
            diff = (predadc - self.dataset.adcdat)
            return tf.reduce_mean((diff*self.dataset.phidat)**2)
        
        def fpetmseloss(nn):
            # assuming mu ~ pet
            upred = nn(self.dataset.xdat)
            predpet = self.param['m']* upred
            diff = (predpet - self.dataset.petdat)
            return tf.reduce_mean((diff*self.dataset.phidat)**2)
        
        

        def fadcnlmseloss(nn):
            # adc nonlinear relation: a = 1 / (1 + 4 m u)
            upred = nn(self.dataset.xdat)
            predadc = 1.0/(1.0 + self.param['m'] * 4* upred)
            diff = (predadc - self.dataset.adcdat) * self.dataset.mask
            return tf.reduce_mean((diff*self.dataset.phidat)**2)
        
        def fadccorloss(nn):
            # correlation of u and adc_data, minimize correlation, negtively correlated
            upred = nn(self.dataset.xdat)
            loss =  tfp.stats.correlation(upred*self.dataset.phidat, self.dataset.adcdat*self.dataset.phidat)
            loss = tf.squeeze(loss)
            return loss
        
        def fmregloss(nn):
            # return (self.param['m']-1.0)**2
            return tf.nn.relu(self.param['m']-1.0) + tf.nn.relu(-self.param['m'])


        def area(upred,th):
            #estimate area above some threshold, assuming the points are uniformly distributed
            # uth = smoothheaviside(upred, th)
            uth = sigmoidbinarize(upred, th)
            # uth = binarize(upred, th)
            return tf.reduce_mean(uth)

        def farea1loss(nn):
            upred = nn(self.dataset.xdat)
            a = area(upred, self.dataset.seg[0,0])
            return (a - self.dataset.area[0,0])**2
        
        def farea2loss(nn):
            upred = nn(self.dataset.xdat)
            a = area(upred, self.dataset.seg[0,1])
            return (a - self.dataset.area[0,1])**2

        # loss function of data, difference of phi * u
        def fdatloss(nn): 
            upred = nn(self.dataset.xdat)
            loss = tf.math.reduce_mean(tf.math.square((self.dataset.udat - upred)*self.dataset.phidat))
            return loss

        # proliferation loss
        def fplfmseloss(nn):
            upred = nn(self.dataset.xdat)
            prolif = 4 * upred * (1-upred)

            loss = tf.math.reduce_mean(tf.math.square((self.dataset.plfdat - prolif)*self.dataset.phidat))
            return loss
        
        #  boundary condition loss
        def bcloss(nn):
            upredbc = nn(self.dataset.xbc)
            loss = tf.math.reduce_mean(tf.math.square((self.dataset.ubc - upredbc)*self.dataset.phibc))
            return loss

        def negloss(nn):
            # penalize negative of u
            upred = nn(self.dataset.xdat)
            neg_loss = tf.reduce_mean(tf.nn.relu(-upred)**2)
            return neg_loss

        def fplfcorloss(nn):
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
        
        # L2 loss
        def fresloss(nn):
            r = pde(self.dataset.xr, nn)
            r2 = tf.math.square(r)
            return tf.reduce_mean(r2)
        
        def fresl1loss(nn):
            r = pde(self.dataset.xr, nn)
            r2 = tf.math.abs(r) # L1 norm
            return tf.reduce_mean(r2)
        
        HUBER_DELTA = 0.001
        def huberloss(y):
            x = tf.math.abs(y)
            x = tf.where(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
            return  tf.reduce_mean(x)

        def freshuberloss(nn):
            r = pde(self.dataset.xr, nn)
            return huberloss(r)

        def fresdtloss(nn):
            # compute residual by evalutaing at discrete time
            nrow = self.dataset.xr.shape[0]
            N = 11
            r2 = np.zeros((nrow,1))
            for t in np.linspace(0.0,1.0,N):
                self.dataset.xr[:,0:1] = t
                r = pde(self.dataset.xr, nn)
                r2 += r**2
            return tf.reduce_mean(r2)/N
        
        def frest0loss(nn):
            # compute residual at time 0
            r = pde(self.dataset.xr0, nn)
            return tf.reduce_mean(r**2)

        reg = None
        if self.opts.get('reg') is not None:
            reg = tf.keras.regularizers.L2(self.opts['reg'])

        self.model = PINN(param=self.param,
                input_dim=self.dim,
                **(self.opts['nn_opts']),
                output_transform=ot,
                regularizer=reg)


        # flosses = {'res': fresloss, 'gradcor': fgradcorloss ,'bc':bcloss, 'cor':fplfcorloss, 'dat': fdatloss, 'dice1':fdice1loss,'dice2':fdice2loss,'area1':farea1loss,'area2':farea2loss, 'pmse': fplfmseloss, 'adc':fadcmseloss}
        flosses = {'res': fresloss, 'reshuber': freshuberloss, 'resl1': fresl1loss,
         'bc':bcloss, 'dat': fdatloss,'adccor': fadccorloss,'resdt':fresdtloss,'rest0':frest0loss,
        'area1':farea1loss,'area2':farea2loss,
        'dice1':fdice1loss,'dice2':fdice2loss,
        'seg1':fseg1loss,'seg2':fseg2loss,
        'seglower1':fseglower1loss,'seglower2':fseglower2loss,
        'adcmse':fadcmseloss, 'adcnlmse':fadcnlmseloss, 
        'plfmse':fplfmseloss, 'plfcor':fplfcorloss,'petmse':fpetmseloss,'mreg':fmregloss}
        
        ftest = {'test':ftestloss} 
        
        if not hasattr(self.dataset,'xtest'):
            self.dataset.xtest = None
            ftest = None

        # Initilize PINN solver
        self.solver = PINNSolver(self.model, pde, 
                                flosses,
                                ftest,
                                xr = self.dataset.xr,
                                xdat = self.dataset.xdat,
                                xtest = self.dataset.xtest,
                                options = opts)

        if opts.get('exactfwd') == True:
            self.param['rD'].assign(self.dataset.rDe)
            self.param['rRHO'].assign(self.dataset.rRHOe)

        if opts.get('resetparam') == True:
            self.param['M'].assign(self.opts['M0'])
            self.param['m'].assign(self.opts['m0'])

    def solve(self):
        if self.opts["num_init_train"] > 0:
            self.solver.solve_with_TFoptimizer(self.optim, N=self.opts["num_init_train"], patience = self.opts["patience"])

        if self.opts['lbfgs_opts'] is not None:
            results = self.solver.solve_with_ScipyOptimizer(method='L-BFGS-B', options=self.opts['lbfgs_opts'])
        
        self.saveopts()
        
    def saveopts(self):
        # save all options 
        z = self.opts | self.solver.info
        tensor2numpy(z)
        fpath = os.path.join(self.opts['model_dir'],'options.json')
        json.dump( z, open( fpath, 'w' ), indent=4 )
    