import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
from weight import Weighting
from util import *


glob_smoothwidth = 20.0
glob_heaviside = 'sigmoid'

def binarize(x, th):
    # https://stackoverflow.com/questions/37743574/hard-limiting-threshold-activation-function-in-tensorflow
    # return 1.0 if x > th
    # this might give None gradient!
    cond = tf.greater(x, th)
    out = tf.where(cond, 1.0, 0.0) 
    return out

def sigmoid_binarize(x, th):
    # smooth heaviside function using a sigmoid
    K = glob_smoothwidth
    return tf.math.sigmoid(K*(x-th))


def double_logistic_sigmoid(x, th):
    # smooth heaviside function using a sigmoid
    z = x-th
    sigma2 = 0.05
    return 0.5 + 0.5 * tf.math.sign(z) * (1.0 - tf.exp(-z**2/sigma2))

def smootherstep_binarize(x, th):
    # smooth heaviside function
    # tfp.math.smootherstep S(x), y goes from 0 to 1 as x goes from 0 1
    # F(x) = S((x+1)/2) = S(x/2+1/2), -1 to 1
    # G(x) = S(Kx/2+1/2), -1/K to 1/K
    K = glob_smoothwidth
    return tfp.math.smootherstep(K * (x-th)/2.0 + 1.0/2.0)

def smoothheaviside(u, th):
    if glob_heaviside == 'sigmoid':
        uth = sigmoid_binarize(u, th)
    else:
        uth = smootherstep_binarize(u, th)
    return uth
        
def mse(x,y,w=1.0):
    return tf.reduce_mean((x-y)**2 *w)

def phimse(x,y,phi):
    return tf.reduce_mean(((x-y)*phi)**2)

def relumse(x):
    # if x<0, 0, otherwise 0.5x^2
    return tf.reduce_mean(0.5*tf.nn.relu(x)**2)
    
def dice(T, P):
    # x is prediction (pos, neg), y is label,
    TP = tf.reduce_sum(T*P)
    FP = tf.reduce_sum((1-T)*P)
    FN = tf.reduce_sum(T*(1-P))
    return 2 * TP / (2*TP + FP + FN)

def diceloss(upred, udat, phi, th):
    pu = smoothheaviside(upred, th)
    d = dice(pu, udat)
    return 1.0-d

def segmseloss(upred, udat, phi, th):    
    '''spatial segmentation loss by mse'''
    uth = smoothheaviside(upred,th)
    return phimse(uth, udat, phi)

def areamseloss(upred, udat, phi, th):    
    '''spatial segmentation loss by area (estimated as ratio of points above threshold)'''
    upred_th = smoothheaviside(upred,th)
    upred_area = tf.reduce_mean(upred_th)

    udat_th = smoothheaviside(udat,th)
    udat_area = tf.reduce_mean(udat_th)
    return mse(upred_area, udat_area)

def loglikely(alpha, y):
    # equation 4, Jana 
    # negative log-likelihood
    P = tf.pow(alpha, y)*tf.pow(1 - alpha, 1.0 - y)
    return - tf.reduce_mean(tf.math.log(P))

def area(upred,th):
    #estimate area above some threshold, assuming the points are uniformly distributed
    uth = smoothheaviside(upred, th)
    return tf.reduce_mean(uth)

def relusqr(p, a, b):
    '''Square of relu function. Penalize out of range'''
    return tf.nn.relu(a-p)**2 + tf.nn.relu(p-b)**2

# HUBER_DELTA = 0.001
# def huberloss(y):
#     x = tf.math.abs(y)
#     x = tf.where(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
#     return  tf.reduce_mean(x)

    
class Losses():
    def __init__(self, model, geomodel, pde, dataset, param, opts):
        self.model = model
        self.geomodel = geomodel
        self.pde = pde
        self.dataset = dataset
        self.param = param
        self.opts = opts
        
        # global variables
        global glob_heaviside
        global glob_smoothwidth
        glob_smoothwidth = self.opts['smoothwidth']
        glob_heaviside = self.opts['heaviside']

        self.idattrain = np.arange(self.opts['Ndat'])
        self.idattest = np.arange(self.opts['Ndat'], self.opts['Ndat'] + self.opts['Ndattest'])
        self.idatall = np.arange(self.opts['Ndat'] + self.opts['Ndattest'])

        self.irestrain = np.arange(self.opts['N'])
        self.irestest = np.arange(self.opts['N'], self.opts['N'] + self.opts['Ntest'])
        self.iresall = np.arange(self.opts['N'] + self.opts['Ntest'])
        
        self.istrain = True
        # index for residual loss and data loss
        self.ires = None
        self.idat = None
        

        self.pdeterm = None
        self.upredxdat = None
        self.upredxr = None
        
        # set data source 
        if self.opts['weights']['udat'] is not None:
            if self.opts['udatsource'] == 'char':
                print('use char uchar_dat, uxrchar\n')
                self.dataset.u_dat = getattr(self.dataset, 'uchar_dat')
                self.dataset.u_res = getattr(self.dataset, 'uchar_res')
                self.dataset.rDe = 1.0
                self.dataset.rRHOe = 1.0
            else:
                print('use gt udat uxr\n')
                self.dataset.u_dat = getattr(self.dataset, 'u_dat')
                self.dataset.u_res = getattr(self.dataset, 'u_res')
        
        # compute testing loss
        self.hastest = False
        if self.opts['Ntest'] > 0:
            self.hastest = True

        
        if self.opts['adcmask'] is not None and hasattr(self.dataset, self.opts['adcmask']):
            # mask for data loss
            self.adcmask = getattr(self.dataset, self.opts['adcmask'])
            print(f"use {self.opts['adcmask']} as mask for adc data loss")
        else:
            self.adcmask = 1.0        

        self.weighting = Weighting(self.opts['weights'], **self.opts['weightopt'])
        

        mregloss = lambda: relusqr(self.param['m'], self.opts['mrange'][0], self.opts['mrange'][1])
        rDregloss = lambda: relusqr(self.param['rD'],  self.opts['rDrange'][0], self.opts['rDrange'][1])
        rRHOregloss = lambda: relusqr(self.param['rRHO'], self.opts['initparam']['rRHO'] * 0.75, self.opts['initparam']['rRHO'] * 1.25)
        Aregloss = lambda: relusqr(self.param['A'], 0.0, 1.0)
        th1regloss = lambda: relusqr(self.param['th1'], 0.3, 0.5)
        th2regloss = lambda: relusqr(self.param['th2'], self.opts['th2range'][0], self.opts['th2range'][1])
        kadcregloss = lambda: relusqr(self.param['kadc'], 0.5, 1.5)


        self.lossdict = {'res':self.resloss, 'resl1':self.resl1loss, 'udat':self.fdatloss, 'bc':self.bcloss,
                         'uxr':self.uxrloss,
                         'udatpos': self.udatpos,
                         'seg1': self.fseg1loss , 'seg2': self.fseg2loss, 
                         'area1': self.farea1loss , 'area2': self.farea2loss, 
                         'dice1': self.fdice1loss , 'dice2': self.fdice2loss, 
                        'petmse': self.fpetmseloss,
                        'adcmse': self.fadcmseloss,
                        'geomse': self.geomseloss,
                        'mreg': mregloss, 'rDreg':rDregloss, 'rRHOreg':rRHOregloss, 'Areg':Aregloss,
                        'th1reg':th1regloss, 'th2reg':th2regloss, 'kadcreg':kadcregloss,
                        'ic': self.icloss,
                        }

        # all training losses
        self.all_losses = self.weighting.weight_keys + ['total'] 
        # all testing loss, exclude reg loss
        self.all_test_losses = [x for x in self.all_losses if 'reg' not in x]
        # all losses exclude residual loss
        self.data_test_loss = [x for x in self.weighting.weight_keys if ('res' not in x and 'bc' not in x)]

        
    def trainmode(self):
        self.istrain = True
        self.ires = self.irestrain
        self.idat = self.idattrain
        
    
    def testmode(self):
        self.istrain = False
        self.ires = self.irestest
        self.idat = self.idattest
    
    def savemode(self):
        self.istrain = False
        self.ires = self.iresall
        self.idat = self.idatall
        
    # evaluate upred at xdat
    # this is saved because upredxdat is used in multiple losses
    def getupredxdat(self):
        self.upredxdat = self.model(self.dataset.X_dat[self.idat,:])

    def getupredxr(self):
        self.upredxr = self.model(self.dataset.X_res[self.ires,:])
    

    # use tf.print properly
    # https://towardsdatascience.com/using-tf-print-in-tensorflow-aa26e1cff11e
    @tf.function
    def getloss(self):
        # compute train or test loss, depending on mode
        self.pdeterm = self.pde.getres(self.dataset, self.ires)
        self.getupredxdat()

        wlosses = {} # dict of weighted loss
        total = 0.0
        for key in self.weighting.weight_keys:
            f = self.lossdict[key] # get loss function
            wlosses[key] = f() # eval loss
            # tf.print(f"{key}: {self.weighting.alphas[key]} {wlosses[key]}")
            total += tf.stop_gradient(self.weighting.alphas[key]) * wlosses[key]
            
        wlosses['total'] = total
        return wlosses

    # def getpdeterm(self):
    #     if self.dataset.xdim == 2:
    #         if self.geomodel is None:
    #             self.pdeterm = self.pde(self.dataset.X_res[self.ires,:], self.model, self.dataset.phiq[self.ires,:], self.dataset.Pq[self.ires,:], self.dataset.DxPphi[self.ires,:], self.dataset.DyPphi[self.ires,:])
    #         else:
    #             self.pdeterm = self.pde(self.dataset.X_res[self.ires,:], self.model, self.geomodel)
    #     else:
    #         self.pdeterm = self.pde(self.dataset.X_res[self.ires,:], self.model, self.dataset.phiq[self.ires,:], self.dataset.Pq[self.ires,:], self.dataset.DxPphi[self.ires,:], self.dataset.DyPphi[self.ires,:], self.dataset.DzPphi[self.ires,:])

    # segmentation mse loss, mse of threholded u and data (patient geometry)
    def fseg1loss(self): 
        return segmseloss(self.upredxdat, self.dataset.u1_dat[self.idat,:], self.dataset.phi_dat[self.idat,:], self.param['th1'])

    def fseg2loss(self): 
        return segmseloss(self.upredxdat, self.dataset.u2_dat[self.idat,:], self.dataset.phi_dat[self.idat,:], self.param['th2'])
    
    # segmentation area loss, mse of ratio above threshold
    def farea1loss(self): 
        return areamseloss(self.upredxdat, self.dataset.u1_dat[self.idat,:], self.dataset.phi_dat[self.idat,:], self.param['th1'])

    def farea2loss(self): 
        return areamseloss(self.upredxdat, self.dataset.u2_dat[self.idat,:], self.dataset.phi_dat[self.idat,:], self.param['th2'])

    def fdice1loss(self): 
        return diceloss(self.upredxdat, self.dataset.u1_dat[self.idat,:], self.dataset.phi_dat[self.idat,:], self.param['th1'])

    def fdice2loss(self): 
        return diceloss(self.upredxdat, self.dataset.u2_dat[self.idat,:], self.dataset.phi_dat[self.idat,:], self.param['th2'])

    def fpetmseloss(self):
        # assuming mu ~ pet
        phiupred = self.upredxdat * self.dataset.phi_dat[self.idat,:]
        predpet = self.param['m']* phiupred - self.param['A']
        return mse(predpet, self.dataset.pet_dat[self.idat,:], w = self.dataset.petseg_dat[self.idat,:])

    def fadcmseloss(self):
        # assuming normalized adc = 1 - k * u
        phiupred = self.upredxdat * self.dataset.phi_dat[self.idat,:]
        predadc = 1.0 - self.param['kadc'] * phiupred
        return mse(predadc, self.dataset.adc_dat[self.idat,:], w = self.adcmask[self.idat,:])
        
    
    def fdatloss(self):
        '''mse of u at Xdat'''
        return phimse(self.dataset.u_dat[self.idat,:], self.upredxdat, self.dataset.phi_dat[self.idat,:])
    
    def udatpos(self):
        '''keep u positive at Xdat'''
        penalty = tf.reduce_mean(tf.nn.relu(-self.upredxdat * self.dataset.phi_dat[self.idat,:])**2)
        return penalty

    def icloss(self):
        '''mse of u at xinit'''
        uinitpred = self.model(self.dataset.X_init[self.ires,:])
        return phimse(self.dataset.u_init[self.ires,:], uinitpred, self.dataset.phiinit[self.ires,:])

    def uxrloss(self):
        '''mse of u at Xr'''
        upredxr = self.model(self.dataset.X_res[self.ires,:])
        return phimse(self.dataset.u_res[self.ires,:], upredxr, self.dataset.phi_res[self.ires,:])
    
    

    #  boundary condition loss
    def bcloss(self):
        # separate xbc
        upredbc = self.model(self.dataset.X_bc[self.ires,:])
        loss = mse(self.dataset.ubc[self.ires,:], upredbc)
        return loss


    def resloss(self):
        r2 = tf.math.square(self.pdeterm['residual'])
        return tf.reduce_mean(r2)

    def resl1loss(self):
        r1 = tf.math.abs(self.pdeterm['residual']) # L1 norm
        return tf.reduce_mean(r1)
    
    def geomseloss(self):
        # loss for geometry
        geo = self.geomodel(self.dataset.X_dat[self.idat,1:])
        return mse(geo['Pwm'],self.dataset.Pwmdat[self.idat,:]) + mse(geo['Pgm'],self.dataset.Pgmdat[self.idat,:]) + mse(geo['phi'],self.dataset.phi_dat[self.idat,:])



# Some loss functions that are not used when rewriting Losses class
'''
# flosses = {'res': fresloss, 'gradcor': fgradcorloss ,'bc':bcloss, 'cor':fplfcorloss, 'dat': fdatloss, 'dice1':fdice1loss,'dice2':fdice2loss,'area1':farea1loss,'area2':farea2loss, 'pmse': fplfmseloss, 'adc':fadcmseloss}
flosses = {'res': fresloss, 'reshuber': freshuberloss, 'resl1': fresl1loss, 'resl1t1': fresl1t1loss,
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

ftest = {'dattest':fdattestloss,'restest':frestestloss} 

def freshuberloss():
    r = pde(self.dataset.X_res, self.model)
    return huberloss(r)

def fresdtloss():
    # compute residual by evalutaing at discrete time
    nrow = self.dataset.X_res.shape[0]
    N = 11
    r2 = np.zeros((nrow,1))
    for t in np.linspace(0.0,1.0,N):
        self.dataset.X_res[:,0:1] = t
        r = pde(self.dataset.X_res, self.model)
        r2 += r**2
    return tf.reduce_mean(r2)/N

def frest0loss():
    # compute residual at time 0
    r = pde(self.dataset.X_res0, self.model)
    return tf.reduce_mean(r['residual']**2)

def fresl1t1loss():
    # compute residual at time 1
    r = pde(self.dataset.X_rest1[self.restrainidx,:], self.model)
    return tf.reduce_mean(tf.math.abs(r['residual']))

maximize dice, minimize 1-dice
def fdice2loss(): 
    upred = self.model(self.dataset.X_dat)
    pu2 = sigmoid_binarize(upred, self.param['th2'])
    d2 = dice(self.dataset.u2_dat, pu2)
    return 1.0-d2


def fseglower1loss(): 
    # if upred>u1, no loss, otherwise, mse loss
    upred = self.model(self.dataset.X_dat)
    diff = self.dataset.phi_dat*(self.dataset.u1_dat * self.param['th1'] - upred)
    return  relumse(diff)

def fseglower2loss(): 
    # if upred>u1, no loss, otherwise, mse loss
    upred = self.model(self.dataset.X_dat)
    diff = self.dataset.phi_dat*(self.dataset.u2_dat * self.param['th2'] - upred)
    return  relumse(diff)

def negloss():
    # penalize negative of u
    upred = self.model(self.dataset.X_dat)
    neg_loss = tf.reduce_mean(tf.nn.relu(-upred)**2)
    return neg_loss

def fplfcorloss():
    # correlation of proliferation 4u(1-u)
    upred = self.model(self.dataset.X_dat)
    # prolif = 4 * upred * (1-upred)
    # loss =  - tfp.stats.correlation(prolif*self.dataset.phi_dat, self.dataset.plfdat*self.dataset.phi_dat)

    loss =  - tfp.stats.correlation(self.dataset.petdat*self.dataset.phi_dat, upred*self.dataset.phi_dat)
    loss = tf.squeeze(loss)
    return loss

def fgradcorloss():
    # correlation of gradient, assume 2D
    u, ux, uy = grad(self.dataset.X_dat, self.model)
    dxprolif = 4 * (ux-2*u*ux)
    dyprolif = 4 * (uy-2*u*uy)
    loss =  - tfp.stats.correlation(dxprolif, self.dataset.dxplfdat) - tfp.stats.correlation(dyprolif, self.dataset.dyplfdat) 
    loss = tf.squeeze(loss)
    return loss

def flike1loss():
    # minimize 1-likelihood, equation 4 Lipkova personalized ...
    upred = self.model(self.dataset.X_dat) *self.dataset.phi_dat
    # alpha = double_logistic_sigmoid(upred, self.param['th1'])
    alpha = sigmoid_binarize(upred, self.param['th1'])
    return loglikely(alpha, self.dataset.u1_dat)

def flike2loss():
    # minimize 1-likelihood, equation 4 Lipkova personalized ...
    upred = self.model(self.dataset.X_dat) *self.dataset.phi_dat
    # alpha = double_logistic_sigmoid(upred, self.param['th2'])
    alpha = sigmoid_binarize(upred, self.param['th2'])
    return loglikely(alpha, self.dataset.u2_dat)



def fadcmseloss():
    # error of adc prediction,
    # this adc is ratio w.r.t characteristic adc
    upred = self.model(self.dataset.X_dat)
    predadc = (1.0 - self.param['m']* upred)
    diff = (predadc - self.dataset.adcdat)
    return tf.reduce_mean((diff*self.dataset.phi_dat)**2)


def fadcnlmseloss():
    # adc nonlinear relation: a = 1 / (1 + 4 m u)
    upred = self.model(self.dataset.X_dat)
    predadc = 1.0/(1.0 + self.param['m'] * 4* upred)
    diff = (predadc - self.dataset.adcdat) * self.dataset.mask
    return tf.reduce_mean((diff*self.dataset.phi_dat)**2)

def fadccorloss():
    # correlation of u and adc_data, minimize correlation, negtively correlated
    upred = self.model(self.dataset.X_dat)
    loss =  tfp.stats.correlation(upred*self.dataset.phi_dat, self.dataset.adcdat*self.dataset.phi_dat)
    loss = tf.squeeze(loss)
    return loss


def farea1loss():
    phiupred = self.model(self.dataset.X_dat) *self.dataset.phi_dat
    a = area(phiupred, self.param['th1'])
    return (a - self.dataset.area[0,0])**2

def farea2loss():
    phiupred = self.model(self.dataset.X_dat) *self.dataset.phi_dat
    a = area(phiupred, self.param['th2'])
    return (a - self.dataset.area[0,1])**2
    
# proliferation loss
def fplfmseloss():
    upred = self.model(self.dataset.X_dat)
    prolif = 4 * upred * (1-upred)

    loss = tf.math.reduce_mean(tf.math.square((self.dataset.plfdat - prolif)*self.dataset.phi_dat))
    return loss

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
'''