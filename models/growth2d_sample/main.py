from growth2d import *


# test 
# hp["num_init_train"] = 100
# lbfgs_opt["maxiter"] = 100

xr,wr = sample_uniform_ball(hp['n_res_pts'], XDIM, BD)

hp['model_name'] = 'plain'
hp['model_dir'] = './'+hp['model_name']

model = PINN(param=param,
            num_hidden_layers=hp["num_hidden_layer"], 
            num_neurons_per_layer=hp["num_hidden_unit"],
            output_transform=output_transform)
model.build(input_shape=(None,DIM))

solver = PINNSolver(model, pde, xr = xr, wr = wr, xtest = xtest, utest = utest, options=hp)


optim = tf.keras.optimizers.Adam(learning_rate=1e-3)
solver.solve_with_TFoptimizer(optim, N=hp["num_init_train"],patience = hp["patience"])

# Solve wit bfgs
results = solver.solve_with_ScipyOptimizer(method='L-BFGS-B', options=lbfgs_opt)

solver.save(header='t x y u w')