import tensorflow as tf
import numpy as np
import sys

class Weighting(object):
    def __init__(self, 
                weights,
                method = 'constant',
                mean_sort = 'full',
                mean_decay_param = 0
                ):

        self.method = method
        self.mean_decay = True if mean_sort == 'decay' else False
        self.mean_decay_param = mean_decay_param

        self.current_iter = tf.zeros((1,))
        self.num_losses = 0
        self.weight_keys = []
        
        self.alphas = []
        for key in weights:
            if weights[key] is not None:
                self.weight_keys.append(key)
                self.num_losses += 1
                self.alphas.append(weights[key])
        
        self.unweighted_losses = tf.Variable(tf.zeros(self.num_losses), trainable=False)
        
        self.alphas = tf.convert_to_tensor(self.alphas)

        if method == 'cov':
            self.alphas = tf.ones((self.num_losses,))/self.num_losses
        
        self.running_mean_L = tf.zeros((self.num_losses,))
        self.running_mean_l = tf.zeros((self.num_losses,))
        self.running_S_l = tf.zeros((self.num_losses,))
        self.running_std_l = None
    
    def weighted_loss(self, unweighted_loss):
        # different ways to update loss
        if self.method == 'cov':
            return self.cov_update(unweighted_loss)
        return self.constant_update(unweighted_loss)
        

    def constant_update(self, unweighted_loss):
        # the weights alpha are constant
        total_loss  = 0.0
        for i, k in enumerate(self.weight_keys):
            total_loss += self.alphas[i] * unweighted_loss[k]
        return total_loss

    
    def cov_update(self, dict_unw_loss):
        
        # unweighted loss, collect alpha as list in same order
        for i, k in enumerate(self.weight_keys):
            self.unweighted_losses[i].assign(dict_unw_loss[k])
        
        L = self.unweighted_losses

        # equation 3. L0 is mu_L_{t-1}. L is L_{t}
        # If we are at the zero-th iteration, set L0 to L. Else use the running mean.
        L0 = tf.identity(L) if self.current_iter == 0 else self.running_mean_L
        # Compute the loss ratios for the current iteration given the current loss L. 
        l = L / L0

        # equation 4
        # self.running_mean_l = mu_l_{it}
        # self.running_std_l = sigma_l_{it}
        # alphas = alpha_{it}
        # If we are in the first iteration set alphas to be the average
        if self.current_iter >= 1:
            ls = self.running_std_l / self.running_mean_l
            self.alphas = ls / tf.reduce_sum(ls)

        # compute total loss
        total_loss = tf.reduce_sum(tf.multiply(self.alphas, L))
        
        
        # Update alpha 
        # equation 5
        # Apply Welford's algorithm to keep running means, variances of L,l. But only do this throughout
        # training the model.
        # 1. Compute the decay parameter the computing the mean.
        if self.current_iter == 0:
            mean_param = 0.0
        elif self.current_iter > 0 and self.mean_decay:
            mean_param = self.mean_decay_param
        else:
            mean_param = (1. - 1 / (self.current_iter + 1))


        # 2. Update the statistics for l
        x_l = l
        new_mean_l = mean_param * self.running_mean_l + (1 - mean_param) * x_l #equation 6, if t=0 then new_mean_l = x_l = 1
        self.running_S_l += (x_l - self.running_mean_l) * (x_l - new_mean_l) # =0 if t=0
        self.running_mean_l = new_mean_l #= mu_(l_t)

        # The variance is S / (t - 1), but we have current_iter = t - 1
        running_variance_l = self.running_S_l / (self.current_iter + 1) # M_(lt) in equation 7
        self.running_std_l = tf.sqrt(running_variance_l + 1e-8)

        # 3. Update the statistics for L
        x_L = L
        self.running_mean_L = mean_param * self.running_mean_L + (1 - mean_param) * x_L
        tf.print(mean_param)
        tf.print(x_L)
        tf.print(self.current_iter)
        # 
        self.current_iter +=1
        return total_loss



if __name__ == "__main__":
    # test using exp random variable,  
    # L = np.random.exponential(k) exp random with mean k variance k^2,
    # mu_L running_mean_L should be close to k
    # mu_l, l = L/mu_L reaning_mean_l should be close to 1,
    # reaning_std_l should be close to 1, 
    # alpha should be close to 0.5 0.5
    # 0<mean_decay_param<1, for mean_decay_param close to one, the result should be similar to no decay

    n = eval(sys.argv[4])

    weight = {'w1':1.0, 'w2': 1.0, 'w3':None}
    w = Weighting(weight, method=sys.argv[1], mean_sort=sys.argv[2],mean_decay_param=eval(sys.argv[3]))
    
    for i in range(n):
        loss = {'w2': tf.convert_to_tensor(np.random.exponential(1)), 'w1':tf.convert_to_tensor(np.random.exponential(10))}
        loss['total']=w.weighted_loss(loss)
        print(w.running_mean_L)
        print(w.running_mean_l)
        print(w.running_std_l)
        print(w.alphas)
        print(loss)
        