from config import *
import numpy as np
from scipy import interpolate
import tensorflow as tf


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

def sample_time_space(n, xndim, bd, spherical=False):
   
    t = np.random.uniform(0,1,(n,1))
    
    if spherical:
        # not uniform in space, denser at center
        space = sample_spherical(n,xndim) * np.random.uniform(0.0, bd, n).reshape((n,1))

    else:
        space = bd*(2*np.random.uniform(0.0, 1, (n,xndim))-1)
    x = np.hstack((t,space))
    return tf.convert_to_tensor(x,dtype=DTYPE)

def sample(n, bd):
   
    ndim = len(bd)
    cols = []
    for i in range(ndim):
        cols.append(tf.random.uniform([n,1],minval=bd[i][0], maxval=bd[i][1]))
    return tf.concat(cols, axis=1)