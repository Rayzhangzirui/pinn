import sys
sys.path.insert(1, '/home/ziruz16/pinn')

from config import *
from pinn import *

from main_stnpos import *

del model
param_fix = {'rD':tf.Variable(0.0, trainable=False,name='rD'),
         'rRHO': tf.Variable(0.0, trainable=False,name='rRHO')}

model2 = PINN(param=param_fix,
            input_dim=DIM,
            scale = [T,L,L],
            num_hidden_layers=hp["num_hidden_layer"], 
            num_neurons_per_layer=hp["num_hidden_unit"],
            output_transform=output_transform)


model2.load_weights('./mri2d_stnpos/ckpt/ckpt-1')

hp['w_dat'] = 0
hp['file_log'] = True
hp['model_name'] = 'cont'
hp['model_dir'] = './cont'
hp['saveckpt'] = True

solver = PINNSolver(model2, pde, xr = xr, xdat = None, udat = None,
                    xtest=xtest, utest=utest, options=hp)

if __name__ == "__main__":
    solver.solve_with_TFoptimizer(optim, N=hp["num_init_train"],patience = hp["patience"])
    solver.solve_with_ScipyOptimizer(method='L-BFGS-B', options=lbfgs_opt)