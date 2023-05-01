# import tensorflow as tf
import numpy as np
import sys
import tensorflow as tf
from config import *

# https://stackoverflow.com/questions/39166941/real-time-moving-averages-in-python
class StreamingMovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []
        self.sum = 0

    def process(self, value):
        # assuming value is 1d numpy array
        self.values.append(value)
        self.sum += value
        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)
        
        return self.sum / len(self.values)



class Weighting(object):
    def __init__(self, 
                weights0,
                method = 'constant',
                window = 100,
                beta = 0.9, 
                whichloss = 'res',
                factor = 1.0,
                interval = 10,
                ):
        ''' weights0 = initial weights
        '''
        self.method = method
        self.window = window # window size for moving average
        self.beta = beta # decay rate
        self.whichloss = whichloss # which loss to track
        self.factor = factor    # factor to multiply the loss used in trackres
        self.current_iter = 0
        self.num_losses = 0 # number of active losses, not including None
        self.weight_keys = [] # list of active losses
        self.skip_weights = {'mreg','rDreg','rRHOreg','Areg','th1reg','th2reg','Mreg'} #weights to skip
        self.active = True # if False, do not update weights
        self.interval = interval # interval to update weights
        
        # initialization, alpha is dict of loss-weight
        self.alphas = {}
        for key in weights0:
            if weights0[key] is not None :
                self.weight_keys.append(key)
                self.num_losses += 1
                self.alphas[key] = tf.Variable(weights0[key], trainable=False, dtype=DTYPE)
        
        if self.method == 'cov' or self.method == 'decay':
            self.unweighted_losses = np.zeros(self.num_losses)
            self.running_mean_L = np.zeros((self.num_losses,))
            self.running_mean_l = np.zeros((self.num_losses,))
            self.running_S_l = np.zeros((self.num_losses,))
            self.running_std_l = None
        

        if self.method == 'trackres' or self.method == 'start0':
            print(f'moving avaerage with windown{self.window}')
            self.stream = StreamingMovingAverage(self.window)


    def decay_update(self, new_alpha):
        '''update weights using exponential decay'''

        for k in self.weight_keys:
            self.alphas[k].assign(self.beta * self.alphas[k] + (1-self.beta) * new_alpha)

    def update_weights(self, unweighted_loss:dict, grad_stat:dict = None):
        # different ways to update loss
        
        if self.active == False:
            # do not update
            return
        
        if self.current_iter % self.interval == 0:
            # update
            
            if self.method == 'cov' or self.method == 'decay':
                self.cov_update(unweighted_loss)
            
            if self.method == 'trackres':
                self.trackres_update(unweighted_loss)

            if self.method == 'start0':
                self.start0_update(unweighted_loss)
            
            if self.method == "grad":
                self.grad_update(grad_stat)
            
            if self.method == "invd":
                self.inverse_dirichlet_update(grad_stat)
        
        self.current_iter +=1
        return
        

    def trackres_update(self, dict_unw_loss):
        '''keep the data loss magnitude same factor of residual
        '''
        # alpha  =  average of residual loss/ loss average of other loss
        for k in self.weight_keys:
            if k in self.skip_weights or k == self.whichloss:
                # skip some weights,
                continue
            new_alpha = dict_unw_loss[self.whichloss] / dict_unw_loss[k] * self.factor
            self.alphas[k] = self.beta * self.alphas[k] + (1-self.beta) * new_alpha
    
    def start0_update(self, dict_unw_loss):
        '''loss starts at 0, then gradually increase to the ratio between residual and data loss
        '''
        # m_i = 0, l_i = beta m(i-1) + (1-beta) l_i
        L = np.array([dict_unw_loss[k] for k in self.weight_keys])
        Lave = self.stream.process(L)
        itrack = self.weight_keys.index(self.whichloss) #index of loss being tracked, usually residual loss

        for i,k in enumerate(self.weight_keys):
            if k in self.skip_weights:
                # skip some weights,
                continue
            if i != itrack:
                self.alphas[k] = self.beta * self.alphas[k] + (1-self.beta) * Lave[itrack]/ Lave[i]
    
    def cov_update(self, dict_unw_loss):
        sys.error('not implemented')
        # loss CANNOT be negative or zero
        # only update weights
        # unweighted loss, collect alpha as list in same order
        L = np.array([dict_unw_loss[k] for k in self.weight_keys])
        
        # equation 3. L0 is mu_L_{t-1}. L is L_{t}
        # If we are at the zero-th iteration, set L0 to L. Else use the running mean.
        L0 = L if self.current_iter == 0 else self.running_mean_L
        # Compute the loss ratios for the current iteration given the current loss L. 
        l = L / L0

        # equation 4
        # self.running_mean_l = mu_l_{it}
        # self.running_std_l = sigma_l_{it}
        # alphas = alpha_{it}
        # If we are in the first iteration set alphas to be the average
        if self.current_iter <= 1:
            alphas = np.ones((self.num_losses,))/self.num_losses
        else:
            ls = self.running_std_l / self.running_mean_l
            alphas = ls / np.sum(ls)
        
        for i, k in enumerate(self.weight_keys):
            self.alphas[k] = alphas[i]

        # compute total loss
        # total_loss = np.dot(self.alphas, L)
        
        
        # Update alpha 
        # equation 5
        # Apply Welford's algorithm to keep running means, variances of L,l. But only do this throughout
        # training the model.
        # 1. Compute the decay parameter the computing the mean.
        if self.current_iter == 0:
            mean_param = 0.0
        elif self.current_iter > 0 and self.method == 'decay':
            # mean_param = 1-1/t, e.g. mean_param = 0.9 same as t=10
            mean_param = self.beta
        else:
            mean_param = (1. - 1 / (self.current_iter + 1))


        # 2. Update the statistics for l
        x_l = l
        new_mean_l = mean_param * self.running_mean_l + (1 - mean_param) * x_l #equation 6, if t=0 then new_mean_l = x_l = 1
        self.running_S_l += (x_l - self.running_mean_l) * (x_l - new_mean_l) # =0 if t=0
        self.running_mean_l = new_mean_l #= mu_(l_t)

        # The variance is S / (t - 1), but we have current_iter = t - 1
        running_variance_l = self.running_S_l / (self.current_iter + 1) # M_(lt) in equation 7
        self.running_std_l = np.sqrt(running_variance_l + 1e-8)

        # 3. Update the statistics for L
        x_L = L
        self.running_mean_L = mean_param * self.running_mean_L + (1 - mean_param) * x_L
        
    def grad_update(self, grad_stat):
        # update by statistics of grad
        # https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs/blob/master/Helmholtz/Helmholtz2D_model_tf.py
        
        for k in self.weight_keys:
            if k in self.skip_weights or k == self.whichloss:
                # skip some weights,
                continue
            new_alpha = grad_stat[self.whichloss]['max'] /  (self.alphas[k] *grad_stat[k]['mean'])
            self.alphas[k].assign(self.beta * self.alphas[k] + (1-self.beta) * new_alpha)
    
    def inverse_dirichlet_update(self,grad_stat):
        # Inverse Dirichlet weighting enables reliable training of physics informed neural networks
        max_std = tf.reduce_max(tf.stack([grad_stat[k]['std'] for k in self.weight_keys]))
        for k in self.weight_keys:
            new_alpha = max_std/grad_stat[k]['std']
            self.alphas[k].assign(self.beta * self.alphas[k] + (1-self.beta) * new_alpha)
            
        



if __name__ == "__main__":
    # test using exp random variable,  
    # L = np.random.exponential(k) exp random with mean k variance k^2,
    # mu_L running_mean_L should be close to k
    # mu_l, l = L/mu_L reaning_mean_l should be close to 1,
    # reaning_std_l should be close to 1, 
    # alpha should be close to 0.5 0.5
    # 0<mean_decay_param<1, for mean_decay_param close to one, the result should be similar to no decay

    weight = {'w1':1.0, 'w2': 1.0, 'res':1.0, 'w3':None}

    n = eval(sys.argv[1])
    w = Weighting(weight, method=sys.argv[2], param=eval(sys.argv[3]))
    

    for i in range(n):
        loss = {'w2': np.random.exponential(1), 'w1':np.random.exponential(10), 'res':np.random.exponential(1)}
        w.update_weights(loss)
        loss['total'] = 0.0
        for k in w.weight_keys:
            loss['total']+= w.alphas[k] * loss[k]
        
        # print(w.running_mean_L)
        # print(w.running_mean_l)
        # print(w.running_std_l)
        print(w.alphas)
        print(loss)
        