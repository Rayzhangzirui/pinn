#!/usr/bin/env python
# rescale rho and D by characteristic rho and D

import sys
sys.path.insert(1, '/home/ziruz16/pinn')
from scipy.io import savemat

from main_stnpos_cont import *


# make prediction and save
model = model2
ts = np.linspace(0,200,21)/T # actual time
n = xdat.shape[0]
m = len(ts)
# xdatnp = xdat.numpy()
xdatnp = xdat
upred = np.zeros([n,m])
for i in range(m):
    xdatnp[:,0] = ts[i]
    upred[:,i:i+1] = model(xdatnp)

predfile = os.path.join(solver.options['model_dir'],'predxdat.mat')
savemat(predfile,{'ts':ts,'upred':upred,'x':xdatnp[:,1::],'scale':scale})
