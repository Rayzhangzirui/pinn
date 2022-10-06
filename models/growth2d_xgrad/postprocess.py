# %%
import sys
sys.path.insert(1, '/home/ziruz16/pinn')
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from config import *
from util import *

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
model_dir = sys.argv[1]
model = tf.keras.models.load_model(os.path.join(model_dir,'savemodel'))
DIM = 3
BD = 1.1
domain = [[0., 1.],[-BD,BD],[-BD,BD]]
# %%
# prediction at collocation points
inv_dat_file = 'exactu_dim2_n20000.txt'
dat = tf.convert_to_tensor(np.loadtxt(inv_dat_file,delimiter=','),dtype=DTYPE)
x_dat = dat[:,0:DIM]
upxr = model(x_dat)
np.savetxt( os.path.join(model_dir,'./predxr.txt'), upxr)

# %%
# prediction at testing point, uniformly sampled
Ntest = 20000
x_t = sample(Ntest, domain).numpy()

ts = np.linspace(0,1.25,6)
upreds = []

for t in ts:
    x_t[:,0]=t
    upred = model(x_t)
    upreds.append(upred.numpy())

upreddat = np.concatenate([x_t[:,1:DIM],*upreds],axis=1)

# %%
header = 'x0 x1 ' + " ".join('t{:g}'.format(t) for t in ts)
np.savetxt( os.path.join(model_dir,'./predt.txt'), upreddat, header = header, comments = '')


