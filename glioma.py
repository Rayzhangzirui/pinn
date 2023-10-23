#!/usr/bin/env python
# solve fisher-komogorov equation in 2d

from re import S
import sys
sys.path.insert(1, '/home/ziruz16/pinn')



from config import *
from pinn import *
from DataSet import DataSet
from Geonn import Geonn
from losses import Losses

import tensorflow_probability as tfp
import tensorflow_addons as tfa




class PDE:
    def __init__(self, nn, param, geonn, dataset):
        self.nn = nn
        self.geonn = geonn
        self.dataset = dataset
        self.dim = dataset.dim
        self.xdim = self.dataset.xdim
        self.param = param
    

    
    @tf.function
    def pde2d(self, xr, phi, P, DxPphi, DyPphi):
        t = xr[:,0:1]
        x = xr[:,1:2]
        y = xr[:,2:3]
        xr = tf.concat([t,x,y], axis=1)
        
        u =  self.nn(xr)
        
        u_t = tf.gradients(u, t)[0]

        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]

        proliferation = self.param['rRHO'] * self.dataset.RHO * phi * u * ( 1 - u/self.param['M'])

        diffusion = self.param['rD'] * self.dataset.DW * (P *phi * (u_xx + u_yy) + self.dataset.L* DxPphi * u_x + self.dataset.L* DyPphi * u_y)
        residual = phi * u_t - ( diffusion +  proliferation)
        return {'residual':residual, 'proliferation': proliferation, 'diffusion': diffusion, 'phiut':phi * u_t}
    
    @tf.function
    def pde3d(self, x_r, phi, P, DxPphi, DyPphi, DzPphi):
        t = x_r[:,0:1]
        x = x_r[:,1:2]
        y = x_r[:,2:3]
        z = x_r[:,3:4]
        xr = tf.concat([t,x,y,z], axis=1)
        u = self.nn(xr)
        
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        
        u_y = tf.gradients(u, y)[0]
        u_yy = tf.gradients(u_y, y)[0]
        
        u_z = tf.gradients(u, z)[0]
        u_zz = tf.gradients(u_z, z)[0]

        proliferation = self.param['rRHO'] * self.dataset.RHO * phi * u * ( 1 - u/self.param['M'])

        diffusion = self.param['rD'] * self.dataset.DW * (P * phi * (u_xx + u_yy + u_zz) + 
                                                            self.dataset.L* DxPphi * u_x +
                                                            self.dataset.L* DyPphi * u_y +
                                                            self.dataset.L* DzPphi * u_z )
        
        residual = phi * u_t - ( diffusion +  proliferation)
        return {'residual':residual, 'proliferation': proliferation, 'diffusion': diffusion, 'phiut':phi * u_t}

    
    # geometry is represented by neural net
    @tf.function
    def pde2dgeo(self, xr):
        t = xr[:,0:1]
        x = xr[:,1:2]
        y = xr[:,2:3]
        xr = tf.concat([t,x,y], axis=1)
        xxr = tf.concat([x,y], axis=1)

        geo = self.geonn(xxr)
        P = geo['Pwm'] + geo['Pgm']/self.dataset.factor # Pwm + Pgm/factor
        phi = geo['phi']

        u =  self.nn(xr)
        
        u_t = tf.gradients(u, t)[0]

        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]

        u_xx = tf.gradients( phi * P * u_x , x)[0]
        u_yy = tf.gradients( phi * P * u_y , y)[0]

        proliferation =  self.dataset.RHO * phi * u * ( 1 - u/self.param['M'])

        diffusion =  self.dataset.DW * (u_xx + u_yy)
        residual = phi * u_t - ( self.param['rD'] * diffusion +  self.param['rRHO']* proliferation)
        return {'residual':residual, 'proliferation': proliferation, 'diffusion': diffusion, 'phiut':phi * u_t}    
    
    @tf.function
    def pde3dgeo(self, xr):
        t = xr[:,0:1]
        x = xr[:,1:2]
        y = xr[:,2:3]
        z = xr[:,3:4]
        xr = tf.concat([t,x,y,z], axis=1)
        xxr = tf.concat([x,y,z], axis=1)

        geo = self.geonn(xxr)
        P = geo['Pwm'] + geo['Pgm']/self.dataset.factor # Pwm + Pgm/factor
        phi = geo['phi']

        u =  self.nn(xr)
        
        u_t = tf.gradients(u, t)[0]

        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_z = tf.gradients(u, z)[0]

        u_xx = tf.gradients( phi * P * u_x , x)[0]
        u_yy = tf.gradients( phi * P * u_y , y)[0]
        u_zz = tf.gradients( phi * P * u_z , z)[0]

        proliferation =  self.dataset.RHO * phi * u * ( 1 - u/self.param['M'])

        diffusion =  self.dataset.DW * (u_xx + u_yy + u_zz)
        residual = phi * u_t - ( self.param['rD'] * diffusion +  self.param['rRHO']* proliferation)
        return {'residual':residual, 'proliferation': proliferation, 'diffusion': diffusion, 'phiut':phi * u_t}    
    

    def getres(self, dataset, idx=None):
        # get terms of pde as dictionary
        if idx is None:
            idx = tf.range(dataset.xr.shape[0])
            
        if self.xdim == 2:
            if self.geonn is None:
                res = self.pde2d(dataset.xr[idx,:], dataset.phiq[idx,:], dataset.Pq[idx,:], dataset.DxPphi[idx,:], dataset.DyPphi[idx,:])
            else:
                res = self.pde2dgeo(dataset.xr[idx,:])
        
        if self.xdim == 3:
            if self.geonn is None:
                res = self.pde3d(dataset.xr[idx,:], dataset.phiq[idx,:], dataset.Pq[idx,:], dataset.DxPphi[idx,:], dataset.DyPphi[idx,:], dataset.DzPphi[idx,:])
            else:
                # not implemented yet
                res = self.pde3dgeo(dataset.xr[idx,:])

        return res





class Gmodel:
    def __init__(self, opts) -> None:

        self.opts = opts

        tf.random.set_seed(self.opts['seed'])
        np.random.seed(self.opts['seed'])

        self.dataset = DataSet(opts['inv_dat_file'])

        # deprecated, 
        # if self.opts.get('endtime') is not None:
        #     # down sample dataset
        #     t = self.opts.get('endtime')
        #     if t < 0:
        #         t = self.dataset.xdat[0,0]
        #         print('use xdat time point')
        #     print('subsample to t less than', t)
        #     idx = np.argwhere(self.dataset.xr[:,0] <= t).flatten()
        #     self.dataset.subsample(idx)

        if self.opts['seed'] > 0:
            self.dataset.shuffle()
        
        ## down sample dataset
        totalxr = self.opts.get('N') + self.opts.get('Ntest')
        totalxdat = self.opts.get('Ndat') + self.opts.get('Ndattest')
        if totalxdat > self.dataset.xdat.shape[0]:
            print('not enough data xdat, reduce Ndattest')
            self.opts['Ndattest'] = totalxdat - self.opts['Ndat']
        N = min(totalxr, totalxdat)
        var_dat = ['u1','u2','u3','udatchar','xdat','petseg','phidat','petdat']
        var_res = ['phiq', 'xr', 'uxrchar','DxPphi', 'Pgmq', 'DyPphi', 'Pwmq', 'DzPphi', 'Pq']
        var_bc = ['ubc','xbc','phibc']
        # downsample res and bc loss
        self.dataset.downsample(N, var_res + var_bc)
        
        # downsample data loss
        if self.opts.get('balanceSample') is True:
            # balance sample
            label = self.dataset.u1 + self.dataset.u2
            idx = balanced_sample_indices(label, N)
            # subsample data loss
            self.dataset.subsample(idx,var_dat)
            # print mean of u1 and u2 to check, if balanced, should be 2/3 and 1/3
            print('mean u1', np.mean(self.dataset.u1))
            print('mean u2', np.mean(self.dataset.u2))
        else:
            # simple down sample
            print('down sample to %d\n', N)
            self.dataset.downsample(N, var_dat)

        ## end of down sample dataset

        if self.opts.get('useupred') is not None:
            # use upred at xdat from other training
            print('use upred from ', opts['useupred'])
            tmpdataset = DataSet(opts['useupred'])
            self.dataset.xdat = np.copy(tmpdataset.xdat)
            self.dataset.udat = np.copy(tmpdataset.upredxdat)
        
        self.dim = self.dataset.dim
        self.xdim = self.dataset.xdim
        

        # choose learning rate schedule
        schedule_type = self.opts['schedule_type']
        if schedule_type == "Constant":
            learning_rate_schedule = self.opts['learning_rate_opts']['initial_learning_rate']

        elif schedule_type == "Exponential":
            learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(**(opts['learning_rate_opts']))
        
        elif schedule_type == "cycle":
            lr = opts['learning_rate_opts']['initial_learning_rate']
            learning_rate_schedule = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=lr,
                                                                         maximal_learning_rate=lr*10,
                                                                         step_size=opts['learning_rate_opts']['step_size'],
                                                                         scale_fn= lambda x: 1/(2.**(x-1)))
        else:
            raise ValueError("Unsupported schedule_type")


        # choose optimizer
        if opts['optimizer'] == 'adamax':
            self.optim = tf.keras.optimizers.Adamax(learning_rate=learning_rate_schedule)
        elif opts['optimizer'] == 'rmsprop':
            self.optim = tf.keras.optimizers.RMSprop(learning_rate=learning_rate_schedule)
        
        elif opts['optimizer'] == 'sgd':
            self.optim = tf.keras.optimizers.SGD(momentum = 0.0, learning_rate=learning_rate_schedule)
        else:
            self.opts['optimizer'] = 'adam'
            self.optim = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

        
        
        
        self.geomodel = None
        # model for probability
        # input is spatial coordiante, output Pwm, Pgm, phi
        if opts['usegeo'] is True:
            self.geomodel = Geonn(input_dim=self.xdim, **(self.opts['geonn_opts']))
            
            # if trained together, restore from same dire as main model
            # if not trained together, restore from restoregeo
            restore_dir = self.opts['restore']
            if not self.opts['restoregeo']:
                restore_dir = self.opts['restoregeo']
            self.setup_ckpt(self.geomodel, ckptdir = 'geockpt', restore = restore_dir)

            self.geomodel.trainable = self.opts['traingeo']

        # get init from dataset
        if opts['initfromdata'] is True:
            # do not set x0, y0, z0, unsclaed value in dataset
            opts['initparam']['rD'] =  self.dataset.rDe
            opts['initparam']['rRHO'] =  self.dataset.rRHOe
            opts['initparam']['M'] =  self.dataset.M
            if hasattr(self.dataset, 'kadc'):
                opts['initparam']['kadc'] =  self.dataset.kadc
            if hasattr(self.dataset, 'm'):
                opts['initparam']['m'] =  self.dataset.m
            if hasattr(self.dataset, 'A'):
                opts['initparam']['A'] =  self.dataset.A
            if hasattr(self.dataset, 'th1'):
                opts['initparam']['th1'] = self.dataset.th1
                opts['initparam']['th2'] = self.dataset.th2
        
        # set data source 
        if self.opts['weights']['udat'] is not None or self.opts['weights']['uxr'] is not None:
            if self.opts['udatsource'] == 'char':
                print('use char udatchar, uxrchar\n')
                self.dataset.udat = getattr(self.dataset, 'udatchar')
                self.dataset.uxr = getattr(self.dataset, 'uxrchar')
                opts['initparam']['rD'] = 1.0
                opts['initparam']['rRHO'] = 1.0
            elif self.opts['udatsource'] == 'noise':
                print('use noisy udat\n')
                self.dataset.udat = getattr(self.dataset, 'udatnz')
                self.dataset.uxr = []
            elif self.opts['udatsource'] == 'gt':
                print('use gt udat\n')
                self.dataset.udat = getattr(self.dataset, 'udat')
                self.dataset.uxr = getattr(self.dataset, 'uxr')
            else:
                raise ValueError('unsupported udatsource')
        
        # set x0 source
        if self.opts['x0source'] == 'gt':
            print('use gt x0\n')
            opts['initparam']['x0'] = (self.dataset.gtx0[0,0] - self.dataset.x0[0,0])/self.dataset.L
            opts['initparam']['y0'] = (self.dataset.gtx0[0,1] - self.dataset.x0[0,1])/self.dataset.L
            if self.xdim == 3:
                opts['initparam']['z0'] = (self.dataset.gtx0[0,2] - self.dataset.x0[0,2])/self.dataset.L
        else:
            print('use x0 (char)\n')

            
            

        # model for probability
        self.param = {
        'rD':  tf.Variable(opts['initparam']['rD'],    trainable=opts.get('trainD'), dtype = DTYPE, name="rD"),
        'rRHO':tf.Variable(opts['initparam']['rRHO'], trainable=opts.get('trainRHO'),dtype = DTYPE,name="rRHO"),
        'M':   tf.Variable(opts['initparam']['M'],     trainable=opts.get('trainM'),dtype = DTYPE,name="M"),
        'm':   tf.Variable(opts['initparam']['m'],     trainable=opts.get('trainm'),dtype = DTYPE,name="m"),
        'A':   tf.Variable(opts['initparam']['A'],     trainable=opts.get('trainA'),dtype = DTYPE,name="A"),
        'x0':  tf.Variable(opts['initparam']['x0'],    trainable=opts.get('trainx0'),dtype = DTYPE,name="x0"),
        'y0':  tf.Variable(opts['initparam']['y0'],    trainable=opts.get('trainx0'),dtype = DTYPE,name="y0"),
        'th1': tf.Variable(opts['initparam']['th1'],   trainable=opts.get('trainth1'),dtype = DTYPE,name="th1"),
        'th2': tf.Variable(opts['initparam']['th2'],   trainable=opts.get('trainth2'),dtype = DTYPE,name="th2"),
        'kadc': tf.Variable(opts['initparam']['kadc'],   trainable=opts.get('trainkadc'),dtype = DTYPE,name="kadc"),
        }

        self.ix = [[self.param['x0'],self.param['y0']]]

        if self.xdim == 3:
            self.param['z0'] = tf.Variable(opts['initparam']['z0'], trainable=opts.get('trainx0'),dtype = DTYPE,name="z0")
            self.ix = [[self.param['x0'],self.param['y0'],self.param['z0']]]

        
        
        def ic(x):
            L = self.dataset.L
            r2 = tf.reduce_sum(tf.square((x[:, 1:self.dim]-self.ix)*L), 1, keepdims=True) # this is in pixel scale, unit mm, 
            return 0.1*tf.exp(-0.1*r2)*self.param['M']

        
        if opts['ictransform'] == False:
            # without output transform, ic as data loss
            print('no nn ic transformation')
            def ot(x,u):
                return u
        else:
            print('apply nn ic transformation')
            def ot(x,u):
                return u* x[:, 0:1]+ ic(x)

        reg = None
        if self.opts.get('weightreg') is not None:
            print('apply weight regularization')
            reg = tf.keras.regularizers.L2(self.opts['weightreg'])

        # Initilize PINN model
        self.model = PINN(param=self.param,
                input_dim=self.dim,
                **(self.opts['nn_opts']),
                output_transform=ot,
                regularizer=reg)
        
        self.pde = PDE(self.model, self.param, self.geomodel, self.dataset)

        # load model, also change self.param
        self.setup_ckpt(self.model, ckptdir = 'ckpt', restore = self.opts['restore'])
        
        
        for x in {'m','A','th1','th2','kadc'}:
            # set init value for m, A, th1, th2
            val = self.opts['initparam'][x]
            if self.opts['seed']>0:
                # add 20% uniform noise to init value
                val = val + val * 0.2*(2*np.random.rand()-1)
            self.param[x].assign(val)
            
                

        losses = Losses(self.model, self.geomodel, self.pde, self.dataset, self.param, self.opts)
                
        # Initilize PINN solver
        self.solver = PINNSolver(self.model, self.pde, 
                                losses,
                                self.dataset,
                                geomodel = self.geomodel,
                                options = opts)


    def solve(self):
        
        # print options and parameters before training
        print (json.dumps(self.opts, indent=2,cls=MyEncoder,sort_keys=True))
        for vname in self.param:
            print(vname, self.param[vname].numpy(), self.param[vname].trainable)
        # save option
        savedict(self.opts, os.path.join(self.opts['model_dir'],'options.json') )
        

        if self.opts["num_init_train"] > 0:
            self.solver.solve_with_TFoptimizer(self.optim, N=self.opts["num_init_train"])
            

        if self.opts['lbfgs_opts'] is not None:
            results = self.solver.solve_with_ScipyOptimizer(method='L-BFGS-B', options=self.opts['lbfgs_opts'])
            
        # save time info 
        savedict(self.solver.info, os.path.join(self.opts['model_dir'],'solverinfo.json') )
    
    def setup_ckpt(self, model, ckptdir = 'ckpt', restore = None):
         # set up check point
        checkpoint = tf.train.Checkpoint(model)
        manager = tf.train.CheckpointManager(checkpoint, directory=os.path.join(self.opts['model_dir'],ckptdir), max_to_keep=4)
        # manager.latest_checkpoint is None if no ckpt found

        restore_str = self.opts['restore']
        # if self.opts['restore'] is specified as ckpt, restore from there
        if restore_str[-1].isdigit():
            checkpoint.restore(restore_str)
            print("Restored from {}".format(restore_str))
        # if self.opts['restore'] is specified as directory, restore from there
        elif os.path.isdir(restore_str):
            # if the path contain ckpt, restore from there
            # otherwise only model directory is specified, restore from latest
            if ckptdir in restore_str:
                checkpoint.restore(restore_str)
            else:
                prev_manager = tf.train.CheckpointManager(checkpoint, directory=os.path.join(self.opts['restore'],ckptdir), max_to_keep=4) 
                ckptpath = prev_manager.latest_checkpoint
                checkpoint.restore(ckptpath)
                print("Restored from {}".format(ckptpath))

        # if self.opts['restore'] is not specified as directory, restore from the same directory
        # if None, restore from latest, if integer, restore by number
        elif manager.latest_checkpoint is not None:
            # as integer, restore from the same directory
            if self.opts['restore'] == '':
                ckptpath = manager.latest_checkpoint
            elif isinstance(self.opts['restore'],int):
                # restore check point in the same directory by integer, 0 = ckpt-1
                ckptpath = manager.checkpoints[self.opts['restore']]
            else:
                raise ValueError('restore should be empty or integer')

            checkpoint.restore(ckptpath)
            print("Restored from {}".format(ckptpath))
            
        else:
            print("No restore")

        self.model.manager = manager
        self.model.checkpoint = checkpoint
    

    
