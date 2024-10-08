from config import *
import numpy as np
from scipy import interpolate
import tensorflow as tf
from time import time
from scipy.io import loadmat
import collections
import json

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
                
# decorator for timing
def timer(func):
    def wrapper(*args, **kwargs):
        start=time()
        result = func(*args, **kwargs)
        end= time()
        print(func.__name__+ ' took ' + str((end-start)) + ' seconds')
        return result
    return wrapper

def interpsol(filepath, nt, nr, x_dat):
    dat = np.loadtxt(filepath,delimiter=',',skiprows=1)
    t = dat[:,0].reshape(nt,nr,order='F')
    r = dat[:,1].reshape(nt,nr,order='F')
    u = dat[:,2].reshape(nt,nr,order='F')

    r_dat = np.sqrt(np.sum(np.square(x_dat[:,1:2]),axis=1,keepdims=True))
    t_dat = x_dat[:,0:1]
    interp_dat = np.concatenate((t_dat,r_dat),axis=1)

    uinterp = interpolate.interpn((t[:,0],r[0,:]),u,interp_dat,method='splinef2d')

    return tf.convert_to_tensor(uinterp,dtype=DTYPE)

def sample_spherical(npoints, ndim):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec.T

def sample_time_space(n, spacedim, bd, spherical=False, tfinal=1):
   
    t = np.random.uniform(0,1,(n,1))*tfinal
    
    if spherical:
        # not uniform in space, denser at center
        space = sample_spherical(n,spacedim) * np.random.uniform(0.0, bd, n).reshape((n,1))

    else:
        space = bd*(2*np.random.uniform(0.0, 1, (n,spacedim))-1)
    x = np.hstack((t,space))
    return tf.convert_to_tensor(x,dtype=DTYPE)

def sample_uniform_ball(n, xdim, R, fw = None):
    ''' sampling uniform [0,1] in time, uniform in ball of xdim
    fw is the weight function, function of r
    '''
    t = np.random.uniform(0,1,(n,1))
    r = np.random.uniform(0.0, R, (n,1))**(1/xdim)
    x = sample_spherical(n,xdim) * r.reshape((n,1))
    dat = tf.convert_to_tensor(np.hstack((t,x)),dtype=DTYPE)
    w = tf.ones([tf.shape(dat)[0],1])
    if fw is not None:
        w = tf.convert_to_tensor(fw(r),dtype=DTYPE)
    return dat, w 

def sample(n, bd):
    ''' uniform sampling in box bd = [[a0,b0],...[ak,bk]]
    '''
    ndim = len(bd)
    cols = []
    for i in range(ndim):
        cols.append(tf.random.uniform([n,1],minval=bd[i][0], maxval=bd[i][1]))
    return tf.concat(cols, axis=1)

def n2t(x):
    # convert numpy to tensor
    if x is None:
        return None
    return tf.convert_to_tensor(x,dtype=DTYPE)

def t2n(x):
    # convert tensor to numpy
    # if x is dict of tensor convert all tensors to numpy
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, tf.Tensor):
        return x.numpy()
    elif isinstance(x, dict):
        dict_of_numpy_arrays = {}
        for key, tensor in x.items():
            if isinstance(tensor, tf.Tensor):
                dict_of_numpy_arrays[key] = tensor.numpy()
        return dict_of_numpy_arrays
    else:
        raise ValueError("Input must be a TensorFlow tensor or a dictionary of tensors.")


def parsedict(d, *argv):
    # parse argv according to dictionary
    i = 0
    while i < len(argv):
        key = argv[i]
        if key in d:
            if isinstance(d[key],str): 
                d[key] = argv[i+1]
            else:
                # if not string, evaluate
                d[key] = eval(argv[i+1])
        i +=2



# https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
# turn a nested dict to a single dict
def flatten(d, parent_key=''):
    items = []
    for k, v in d.items():
        new_key = k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)

def str_from_dict(d, prefix, keys):
    # generate string from dictionary, as "key1 val1 key2 val2 ..."
    # d is the option dictionary, might be nested dict
    # keys is a list of keys
    # skip None
    flatd = flatten(d)
    s = prefix
    for k in keys:
        if flatd.get(k) is None:
            continue
        s+= ("_"+str(k))
        s+= str(flatd[k])
    return s

def tensor2numpy(d):
    # recursively convert tensor in dictionary to numpy
    for key in d:
        if isinstance(d[key],dict):
            tensor2numpy(d[key])
        if tf.is_tensor(d[key]):
            d[key] = float(d[key])

def savedict(dict, fpath):
        # save all options 
        tensor2numpy(dict)
        json.dump( dict, open( fpath, 'w' ), indent=4, cls=MyEncoder, sort_keys=True)





def balanced_sample_indices(labels, m):
    # sample m indices from labels, with almost equal number of samples from each class
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    k = len(unique_classes)

    samples_per_class = m // k
    balanced_sample_indices = np.array([], dtype=int)

    for label in unique_classes:
        class_indices = np.where(labels == label)[0]
        class_sample_indices = np.random.choice(class_indices, samples_per_class, replace=False)
        balanced_sample_indices = np.concatenate((balanced_sample_indices, class_sample_indices))

    # add any remaining samples
    remaining = m - len(balanced_sample_indices)
    if remaining > 0:
        remaining_indices = np.random.choice(np.arange(len(labels)), remaining, replace=False)
        balanced_sample_indices = np.concatenate((balanced_sample_indices, remaining_indices))

    return balanced_sample_indices