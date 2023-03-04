# import tensorflow as tf
import numpy as np
import sys

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
                weights,
                method = 'constant',
                param = None,
                whichloss = 'res',
                factor = 1.0,
                ):

        self.method = method
        self.param = param
        self.whichloss = whichloss
        self.factor = factor
        self.current_iter = 0
        self.num_losses = 0
        self.weight_keys = []
        self.skip_weights = {'mreg','rDreg','rRHOreg','Areg'} #weights to skip
        
        # initialization
        self.alphas = {}
        for key in weights:
            if weights[key] is not None :
                self.weight_keys.append(key)
                self.num_losses += 1
                self.alphas[key] = weights[key]
        
        if self.method == 'cov' or self.method == 'decay':
            self.unweighted_losses = np.zeros(self.num_losses)
            self.running_mean_L = np.zeros((self.num_losses,))
            self.running_mean_l = np.zeros((self.num_losses,))
            self.running_S_l = np.zeros((self.num_losses,))
            self.running_std_l = None

        if self.method == 'trackres':
            print(f'moving avaerage with windown{self.param}')
            self.stream = StreamingMovingAverage(self.param)


    def update_weights(self, unweighted_loss):
        # different ways to update loss
        if self.method == 'cov' or self.method == 'decay':
            return self.cov_update(unweighted_loss)
        
        if self.method == 'trackres':
            return self.trackres_update(unweighted_loss)
        
        
        
    def trackres_update(self, dict_unw_loss):
        # keep losses same magnitude as residual
        # alpha  =  average of residual loss/ loss average of other loss
        L = np.array([dict_unw_loss[k] for k in self.weight_keys])
        Lave = self.stream.process(L)
        itrack = self.weight_keys.index(self.whichloss) #index of loss being tracked, usually residual loss

        # initially keep constant, then start changing the weight
        if self.current_iter > self.stream.window_size:
            for i,k in enumerate(self.weight_keys):
                if k in self.skip_weights:
                    # skip some weights,
                    continue
                if i != itrack:
                    self.alphas[k] = Lave[itrack]/ Lave[i] * self.factor # weight of res/ weight of loss
        self.current_iter +=1


    
    def cov_update(self, dict_unw_loss):
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
            mean_param = self.param
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
        self.current_iter +=1
        
        



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
        