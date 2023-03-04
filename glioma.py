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
            total = self.opts.get('N') + self.opts.get('Ntest')
            self.dataset.downsample(total)
            self.trainidx = np.arange(self.opts['N'])
            self.testidx = np.arange(self.opts['N'],self.opts['Ntest'])
        if opts.get('useupred') is not None:
            # use upred at xdat from other training
            tmpdataset = DataSet(opts['useupred'])
            self.dataset.xdat = np.copy(tmpdataset.xdat)
            self.dataset.udat = np.copy(tmpdataset.upredxdat)
        
        if opts.get('datmask') is not None:
            # mask for data loss
            self.mask = getattr(self.dataset, opts.get('datmask'))
            print(f"use {opts.get('datmask')} as mask")
        else:
            self.mask = 1.0

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
        
        
        self.geomodel = None
        # model for probability
        # input is spatial coordiante, output Pwm, Pgm, phi
        if opts['usegeo'] is True:
            self.geomodel = tf.keras.Sequential(
                                [
                                    tf.keras.layers.Dense(32, activation="tanh", input_shape=(self.xdim,)),
                                    tf.keras.layers.Dense(32, activation="tanh"),
                                    tf.keras.layers.Dense(3, activation="sigmoid"),
                                ]
                            )

        self.param = {'rD':tf.Variable( opts['D0'], trainable=opts.get('trainD'), dtype = DTYPE, name="rD"),
        'rRHO': tf.Variable(opts['RHO0'], trainable=opts.get('trainRHO'),dtype = DTYPE,name="rRHO"),
        # 'factor': tf.Variable(opts['r0'], trainable=opts.get('trainfactor'),dtype = DTYPE,name="r"),
        'M': tf.Variable(opts['M0'], trainable=opts.get('trainM'),dtype = DTYPE,name="M"),
        'm': tf.Variable(opts['m0'], trainable=opts.get('trainm'),dtype = DTYPE,name="m"),
        'A': tf.Variable(opts['A0'], trainable=opts.get('trainA'),dtype = DTYPE,name="A"),
        'x0': tf.Variable(opts['x0'], trainable=opts.get('trainx0'),dtype = DTYPE,name="x0"),
        'y0': tf.Variable(opts['y0'], trainable=opts.get('trainx0'),dtype = DTYPE,name="y0"),
        'th1': tf.Variable(opts['th1'], trainable=opts.get('trainth1'),dtype = DTYPE,name="th1"),
        'th2': tf.Variable(opts['th2'], trainable=opts.get('trainth2'),dtype = DTYPE,name="th2"),
        }

        self.ix = [[self.param['x0'],self.param['y0']]]

        # for computing residual at initial time
        self.dataset.xr0 = np.copy(self.dataset.xr)
        self.dataset.xr0[:,0] = 0.0

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

        if self.xdim == 2 and self.geomodel is None:
                # geometry is provided by data
                @tf.function
                def pde(xr, nn):
                    t = xr[:,0:1]
                    x = xr[:,1:2]
                    y = xr[:,2:3]
                    xr = tf.concat([t,x,y], axis=1)
                    
                    u =  nn(xr)
                    
                    u_t = tf.gradients(u, t)[0]

                    u_x = tf.gradients(u, x)[0]
                    u_y = tf.gradients(u, y)[0]
                    
                    u_xx = tf.gradients(u_x, x)[0]
                    u_yy = tf.gradients(u_y, y)[0]

                    proliferation = self.param['rRHO'] * self.dataset.RHO * self.dataset.phiq * u * ( 1 - u/self.param['M'])

                    diffusion = self.param['rD'] * self.dataset.DW * (self.dataset.Pq *self.dataset.phiq * (u_xx + u_yy) + self.dataset.L* self.dataset.DxPphi * u_x + self.dataset.L* self.dataset.DyPphi * u_y)
                    residual = self.dataset.phiq * u_t - ( diffusion +  proliferation)
                    return {'residual':residual, 'proliferation': proliferation, 'diffusion': diffusion, 'phiut':self.dataset.phiq * u_t}

        if self.xdim == 2 and self.geomodel is not None:
            # geometry is represented by neural net
                @tf.function
                def pde(xr, nn):
                    t = xr[:,0:1]
                    x = xr[:,1:2]
                    y = xr[:,2:3]
                    xr = tf.concat([t,x,y], axis=1)
                    xxr = tf.concat([x,y], axis=1)

                    geo = self.geomodel(xxr)
                    P = geo[:,0:1] + geo[:,1:2]/self.dataset.factor # Pwm + Pgm/factor
                    phi = geo[:,2:3]

                    u =  nn(xr)
                    
                    u_t = tf.gradients(u, t)[0]

                    u_x = tf.gradients(u, x)[0]
                    u_y = tf.gradients(u, y)[0]

                    u_xx = tf.gradients( phi * P * u_x , x)[0]
                    u_yy = tf.gradients( phi * P * u_y , y)[0]

                    proliferation =  self.dataset.RHO * self.dataset.phiq * u * ( 1 - u/self.param['M'])

                    diffusion =  self.dataset.DW * (self.dataset.Pq *self.dataset.phiq * (u_xx + u_yy) + self.dataset.L* self.dataset.DxPphi * u_x + self.dataset.L* self.dataset.DyPphi * u_y)
                    residual = self.dataset.phiq * u_t - ( self.param['rD'] * diffusion +  self.param['rRHO']* proliferation)
                    return {'residual':residual, 'proliferation': proliferation, 'diffusion': diffusion, 'phiut':self.dataset.phiq * u_t}

            
            
        # else:
            # @tf.function
            # def pde(x_r, f):
                # TODO: need update for 3d residual
                # t,x,y normalized here
                # t = x_r[:,0:1]
                # x = x_r[:,1:2]
                # y = x_r[:,2:3]
                # z = x_r[:,3:4]
                # xr = tf.concat([t,x,y,z], axis=1)
                # u =  f(xr)
                
                # u_t = tf.gradients(u, t)[0]
                # u_x = tf.gradients(u, x)[0]
                # u_xx = tf.gradients(u_x, x)[0]
                
                # u_y = tf.gradients(u, y)[0]
                # u_yy = tf.gradients(u_y, y)[0]
                
                # u_z = tf.gradients(u, z)[0]
                # u_zz = tf.gradients(u_z, z)[0]

                # prolif = self.param['rRHO'] * self.dataset.RHO * self.dataset.phiq * u * ( 1 - u/self.param['M'])
                # diffusion = self.param['rD'] * self.dataset.DW * (self.dataset.Pq *self.dataset.phiq * (u_xx + u_yy + u_zz) 
                # + self.dataset.L* (self.dataset.DxPphi * u_x + self.dataset.DyPphi * u_y + self.dataset.DzPphi * u_z))

                # res = self.dataset.phiq * u_t - ( diffusion +  prolif)
                # return res


        @tf.function
        def grad(X):
            # t,x,y normalized here
            t = X[:,0:1]
            x = X[:,1:2]
            y = X[:,2:3]
            Xcat = tf.concat([t,x,y], axis=1)
            u =  self.model(Xcat)
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
            return tf.math.sigmoid(20*(x-th))
        

        def double_logistic_sigmoid(x, th):
            # smooth heaviside function using a sigmoid
            z = x-th
            sigma2 = 0.05
            return 0.5 + 0.5 * tf.math.sign(z) * (1.0 - tf.exp(-z**2/sigma2))

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
        def fdice1loss(): 
            upred = self.model(self.dataset.xdat)
            pu1 = sigmoidbinarize(upred, self.param['th1'])
            d1 = dice(self.dataset.u1, pu1)
            return 1.0-d1

        def fdice2loss(): 
            upred = self.model(self.dataset.xdat)
            pu2 = sigmoidbinarize(upred, self.param['th2'])
            d2 = dice(self.dataset.u2, pu2)
            return 1.0-d2
        
        # segmentation mse loss, mse of threholded u and data (patient geometry)
        def fseg1loss(): 
            upred = self.model(self.dataset.xdat)
            pu1 = sigmoidbinarize(upred, self.param['th1'])
            # pu1 = smoothheaviside(upred, self.param['th1'])
            return tf.reduce_mean((self.dataset.phidat*(pu1-self.dataset.u1))**2)

        def fseg2loss(): 
            upred = self.model(self.dataset.xdat)
            pu2 = sigmoidbinarize(upred, self.param['th2'])
            # pu2 = smoothheaviside(upred, self.param['th2'])
            return tf.reduce_mean((self.dataset.phidat*(pu2-self.dataset.u2))**2)

        def relumse(x):
            # if x<0, 0, otherwise 0.5x^2
            return tf.reduce_mean(0.5*tf.nn.relu(x)**2)

        def fseglower1loss(): 
            # if upred>u1, no loss, otherwise, mse loss
            upred = self.model(self.dataset.xdat)
            diff = self.dataset.phidat*(self.dataset.u1 * self.param['th1'] - upred)
            return  relumse(diff)

        def fseglower2loss(): 
            # if upred>u1, no loss, otherwise, mse loss
            upred = self.model(self.dataset.xdat)
            diff = self.dataset.phidat*(self.dataset.u2 * self.param['th2'] - upred)
            return  relumse(diff)

        def loglikely(alpha, y):
            # equation 4, Jana 
            # negative log-likelihood
            P = tf.pow(alpha, y)*tf.pow(1 - alpha, 1.0 - y)
            return - tf.reduce_mean(tf.math.log(P))

        def flike1loss():
            # minimize 1-likelihood, equation 4 Lipkova personalized ...
            upred = self.model(self.dataset.xdat) *self.dataset.phidat
            # alpha = double_logistic_sigmoid(upred, self.param['th1'])
            alpha = sigmoidbinarize(upred, self.param['th1'])
            return loglikely(alpha, self.dataset.u1)
        
        def flike2loss():
            # minimize 1-likelihood, equation 4 Lipkova personalized ...
            upred = self.model(self.dataset.xdat) *self.dataset.phidat
            # alpha = double_logistic_sigmoid(upred, self.param['th2'])
            alpha = sigmoidbinarize(upred, self.param['th2'])
            return loglikely(alpha, self.dataset.u2)

        def fadcmseloss():
            # error of adc prediction,
            # this adc is ratio w.r.t characteristic adc
            upred = self.model(self.dataset.xdat)
            predadc = (1.0 - self.param['m']* upred)
            diff = (predadc - self.dataset.adcdat)
            return tf.reduce_mean((diff*self.dataset.phidat)**2)
        
        def fpetmseloss():
            # assuming mu ~ pet
            phiupred = self.model(self.dataset.xdat) * self.dataset.phidat 
            predpet = self.param['m']* phiupred - self.param['A']
            return mse(predpet, self.dataset.petdat, w = self.mask)
        
        def fadcnlmseloss():
            # adc nonlinear relation: a = 1 / (1 + 4 m u)
            upred = self.model(self.dataset.xdat)
            predadc = 1.0/(1.0 + self.param['m'] * 4* upred)
            diff = (predadc - self.dataset.adcdat) * self.dataset.mask
            return tf.reduce_mean((diff*self.dataset.phidat)**2)
        
        def fadccorloss():
            # correlation of u and adc_data, minimize correlation, negtively correlated
            upred = self.model(self.dataset.xdat)
            loss =  tfp.stats.correlation(upred*self.dataset.phidat, self.dataset.adcdat*self.dataset.phidat)
            loss = tf.squeeze(loss)
            return loss
        

        def relusqr(p, a, b):
            # square of relu function
            return tf.nn.relu(a-p)**2 + tf.nn.relu(p-b)**2
        
        # mregloss = lambda: mse(self.param['m'], self.opts['m0'])
        # rDregloss = lambda: mse(self.param['rD'], self.opts['D0'])
        # rRHOregloss = lambda: mse(self.param['rRHO'], self.opts['RHO0'])

        mregloss = lambda: relusqr(self.param['m'], self.opts['mrange'][0], self.opts['mrange'][1])
        rDregloss = lambda: relusqr(self.param['rD'], self.opts['D0'] * 0.1, self.opts['D0'] * 1.9)
        rRHOregloss = lambda: relusqr(self.param['rRHO'], self.opts['RHO0'] * 0.1, self.opts['RHO0'] * 1.9)
        Aregloss = lambda: relusqr(self.param['A'], 0.0, 1.0)

        def area(upred,th):
            #estimate area above some threshold, assuming the points are uniformly distributed
            # uth = smoothheaviside(upred, th)
            uth = sigmoidbinarize(upred, th)
            # uth = binarize(upred, th)
            return tf.reduce_mean(uth)

        def farea1loss():
            phiupred = self.model(self.dataset.xdat) *self.dataset.phidat
            a = area(phiupred, self.param['th1'])
            return (a - self.dataset.area[0,0])**2
        
        def farea2loss():
            phiupred = self.model(self.dataset.xdat) *self.dataset.phidat
            a = area(phiupred, self.param['th2'])
            return (a - self.dataset.area[0,1])**2

        # loss function of data, difference of phi * u
        def fdatloss(): 
            upred = self.model(self.dataset.xdat)
            loss = tf.math.reduce_mean(tf.math.square((self.dataset.udat - upred)*self.dataset.phidat))
            return loss

        # proliferation loss
        def fplfmseloss():
            upred = self.model(self.dataset.xdat)
            prolif = 4 * upred * (1-upred)

            loss = tf.math.reduce_mean(tf.math.square((self.dataset.plfdat - prolif)*self.dataset.phidat))
            return loss
        
        #  boundary condition loss
        def bcloss():
            upredbc = self.model(self.dataset.xbc)
            loss = tf.math.reduce_mean(tf.math.square((self.dataset.ubc - upredbc)*self.dataset.phibc))
            return loss

        def negloss():
            # penalize negative of u
            upred = self.model(self.dataset.xdat)
            neg_loss = tf.reduce_mean(tf.nn.relu(-upred)**2)
            return neg_loss

        def fplfcorloss():
            # correlation of proliferation 4u(1-u)
            upred = self.model(self.dataset.xdat)
            # prolif = 4 * upred * (1-upred)
            # loss =  - tfp.stats.correlation(prolif*self.dataset.phidat, self.dataset.plfdat*self.dataset.phidat)

            loss =  - tfp.stats.correlation(self.dataset.petdat*self.dataset.phidat, upred*self.dataset.phidat)
            loss = tf.squeeze(loss)
            return loss
        
        def fgradcorloss():
            # correlation of gradient, assume 2D
            u, ux, uy = grad(self.dataset.xdat, self.model)
            dxprolif = 4 * (ux-2*u*ux)
            dyprolif = 4 * (uy-2*u*uy)
            loss =  - tfp.stats.correlation(dxprolif, self.dataset.dxplfdat) - tfp.stats.correlation(dyprolif, self.dataset.dyplfdat) 
            loss = tf.squeeze(loss)
            return loss
        
        def ftestloss():
            upred = self.model(self.dataset.xtest)
            loss = tf.math.reduce_mean(tf.math.square((self.dataset.utest - upred)*self.dataset.phitest))
            return loss
        
        # L2 loss
        def fresloss():
            r = pde(self.dataset.xr[self.trainidx,:], self.model)
            r2 = tf.math.square(r['residual'])
            return tf.reduce_mean(r2)
        
        def fresl1loss():
            r = pde(self.dataset.xr[self.trainidx,:], self.model)
            r2 = tf.math.abs(r['residual']) # L1 norm
            return tf.reduce_mean(r2)
        
        HUBER_DELTA = 0.001
        def huberloss(y):
            x = tf.math.abs(y)
            x = tf.where(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
            return  tf.reduce_mean(x)

        def freshuberloss():
            r = pde(self.dataset.xr, self.model)
            return huberloss(r)

        def fresdtloss():
            # compute residual by evalutaing at discrete time
            nrow = self.dataset.xr.shape[0]
            N = 11
            r2 = np.zeros((nrow,1))
            for t in np.linspace(0.0,1.0,N):
                self.dataset.xr[:,0:1] = t
                r = pde(self.dataset.xr, self.model)
                r2 += r**2
            return tf.reduce_mean(r2)/N
        
        def frest0loss():
            # compute residual at time 0
            r = pde(self.dataset.xr0, self.model)
            return tf.reduce_mean(r**2)
        
        def mse(x,y,w=1.0):
            return tf.reduce_mean((x-y)**2 *w)

        def geomseloss():
            P = self.geomodel(self.dataset.xr[:,1:self.xdim+1])
            return mse(P[:,0:1],self.dataset.Pwmq)+mse(P[:,1:2],self.dataset.Pgmq) + mse(P[:,2:3],self.dataset.phiq)

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
        'like1':flike1loss,'like2':flike2loss,
        'adcmse':fadcmseloss, 'adcnlmse':fadcnlmseloss, 
        'plfmse':fplfmseloss, 'plfcor':fplfcorloss,'petmse':fpetmseloss,
        'mreg': mregloss, 'rDreg':rDregloss, 'rRHOreg':rRHOregloss, 'Areg':Aregloss,
        'geomse':geomseloss}
        
        ftest = {'test':ftestloss} 
        
        if not hasattr(self.dataset,'xtest'):
            self.dataset.xtest = None
            ftest = None

        # Initilize PINN solver
        self.solver = PINNSolver(self.model, pde, 
                                flosses,
                                ftest,
                                geomodel = self.geomodel,
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
    