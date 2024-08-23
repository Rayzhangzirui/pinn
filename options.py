import sys
import ast
import json

# default ftol 2.2204460492503131e-09 from  https://github.com/scipy/scipy/blob/v1.10.1/scipy/optimize/_lbfgsb_py.py#L48-L207
# about np.finfo(float).eps*10e6
lbfgs_opts = {"maxcor": 100, 'ftol':2.2204460492503131e-09, 'gtol':0.0, 'maxfun': 10000, "maxiter": 10000, "maxls": 50}

nn_opts = {'num_hidden_layers':4, 'num_neurons_per_layer':64, 'resnet': False, 'userbf' : False, "activation":'tanh',"userff":False}

geonn_opts = {'depth':3, 'width':64}

# option for data set
data_opts = {'Nres': 50000, 'resratio':1.0, 'Ndat': 50000, 'Ndatratio':1.0}

weights = {'res':1.0, 'resl1':None, 'geomse':None, 'petmse': None, 'bc':None, 'udat':None, 
    'plfcor':None, 'uxr':None, 'u0dat':None, 'adcmse':None,
    'udatpos':None,
    'mreg': None, 'rDreg':None, 'rRHOreg':None, 'Areg':None, 'kadcreg':None,
    'ic':None,
    'area1':None, 'area2':None,'seg1':None, 'seg2':None, 'seglower1':None, 'seglower2':None, 'dice1':None, 'dice2':None}

# initial paramter
initparam = {'rD': 1.0, 'rRHO': 1.0, 'M': 1.0, 'm': 1.0, 'th1':0.35, 'th2':0.6, 'A':0.0, 'x0':0.0, 'y0':0.0, 'z0':0.0, 'kadc':1.0}

earlystop_opts = {'patience': 1000, 'min_delta': 1e-6, "monitor":['total'],'burnin':1000}

opts = {
    "seed": 0,
    "tag" : '',
    "note": '',
    "model_dir": '',
    "num_init_train" : 100000, # initial traning iteration
    "ictransform":True,
    "N" : 50000, # number of residual point
    "Ntest":50000,
    "Ndat":50000,
    "Ndattest":50000,
    "balanceSample":False,
    "endtime": 1.0,
    "nn_opts": nn_opts,
    "geonn_opts": geonn_opts,
    "print_res_every" : 20, # print residual
    "save_res_every" : None, # save residual
    "weights" : weights, # weight of data, weight of res is 1
    "ckpt_every": 20000,
    "initfromdata":True,
    "earlystop_opts":earlystop_opts,
    "file_log":True,
    "saveckpt":True,
    "initparam":initparam,
    "adcmask":'u3',
    "smoothwidth": 20,
    "heaviside":'sigmoid',
    "udatsource":'char',
    "x0source":'char',
    "mrange":[0.75,1.25],
    "rDrange":[0.75,1.25],
    "th2range":[0.5,0.8],
    "trainD":True,
    "trainRHO":True,
    "trainM":False,
    "trainm":False,
    "trainA":False,
    "trainth1":False,
    "trainth2":False,
    "trainx0":False,
    "trainkadc":False,
    'inv_dat_file': '',
    "lbfgs_opts":lbfgs_opts,
    "randomt": -1.0,
    "optimizer":'adam',
    "restore": '',
    "restoregeo": '',
    "copyfrom": '',
    "trainnnweight":None,
    "schedule_type":'Constant', #Constant, Exponential
    "learning_rate_opts": {'initial_learning_rate': 0.001, 'decay_rate': 0.01, 'decay_steps':100000,'step_size':2000},
    "useupred": None,
    "gradnorm":False,
    "outputderiv":False,
    "usegeo":False,
    "traingeo":False,
    "patientweight":1.0,
    "simtype":'fitfwd',
    "weightopt": {'method': 'constant','window': 100, 'whichloss': 'res', 'factor':0.001}
    }


def update_nested_dict(nested_dict, key, value):
    """
    Recursively updates a nested dictionary by finding the specified key and assigning the new value to it.
    
    Args:
    - nested_dict (dict): the nested dictionary to update
    - key (str): the key to find and update
    - value (any): the new value to assign to the key
    
    Returns:
    - None
    """
    found =  False
    for k, v in nested_dict.items():
        if k == key:
            nested_dict[k] = value
            found = True
        elif isinstance(v, dict):
            found = found or update_nested_dict(v, key, value)

    return found

def copy_nested_dict(orig_dict, update_dict):
    for key, value in update_dict.items():
        if isinstance(value, dict):
            orig_dict[key] = copy_nested_dict(orig_dict.get(key, {}), value)
        else:
            orig_dict[key] = value
    return orig_dict


def get_nested_dict(nested_dict, target_key):
    """
    get value from nested dict, if not found, result is None
    """
    for k, v in nested_dict.items():
        if k == target_key:
            return v
        elif isinstance(v, dict):
            result = get_nested_dict(v, target_key)
            if result is not None:
                return result





class Options(object):
    def __init__(self, opts = opts):
        self.opts = opts
        

    def parse_args(self, *args):
        
        # initialize by copy from previous simulation
        if 'copyfrom' in args:
            file = args[args.index('copyfrom')+1]
            with open(file, 'r') as f:
                copyopts = json.load(f)
            copy_nested_dict(self.opts, copyopts)
            self.opts['restore'] = self.opts['model_dir']

        # set all weights to None except res
        for k in self.opts['weights'].keys():
            if k != 'res':
                self.opts['weights'][k] = None

        if 'simtype' in args:
            tmp = args[args.index('simtype')+1]
            self.opts['simtype'] = tmp

        
        self.process_simtype()

        # second pass, might modify the dictionary, especially for weights
        self.parse_nest_args(*args)

        self.set_trainable()
        
        # trim the weights
        # self.opts['weights'] = {k: v for k, v in self.opts['weights'].items() if v is not None}
        self.eval_name()
    
    def eval_name(self):
        # evaluate the name of the model
        size = f"{self.opts['nn_opts']['num_hidden_layers']}x{self.opts['nn_opts']['num_neurons_per_layer']}"
        self.opts['tag'] = (self.opts['tag']).format(size=size)

    def parse_nest_args(self, *args):
        # parse args according to dictionary
        
        i = 0
        while i < len(args):
            key = args[i]
            if key in {"simtype","copyfrom"}:
                i += 2
                continue
            default_val = get_nested_dict(self.opts, key)
            if isinstance(default_val,str):
                val = args[i+1]
            
            else:
                try:
                    val = ast.literal_eval(args[i+1])
                except ValueError as ve:
                    print(f'error parsing {args[i]} and {args[i+1]}: {ve}')
                    sys.exit(1)
            
            found = update_nested_dict(self.opts, key, val)
            if not found:
                raise ValueError('Key %s not found in dictionary' % key)
            i +=2

    def process_simtype(self):
        # set some options according to simtype
        simtypes = self.opts['simtype'].split(',')
        
        for simtype in simtypes:
            if simtype == 'solvechar':
                # solve characteristic equation
                self.opts['trainD'] = False
                self.opts['trainRHO'] = False
                self.opts['weights']['res'] = 1.0
                self.opts['earlystop_opts']['monitor'] = ['total','totaltest']
            
            elif simtype == 'fitfwd':
                # use res and dat to learn solution
                self.opts['trainD'] = False
                self.opts['trainRHO'] = False
                self.opts['weights']['res'] = 1.0
                self.opts['weights']['udat'] = 1.0
                self.opts['earlystop_opts']['monitor'] = ['total','totaltest']
            
            elif simtype == 'synthetic':
                # simple synthetic data, infer D and RHO
                self.opts['trainD'] = True
                self.opts['trainRHO'] = True
                self.opts['weights']['res'] = 1.0
                self.opts['weights']['udat'] = 1.0
            
            
            elif simtype == 'patient':
                # patient data or full synthetic data, infer all parameters
                self.opts['trainD'] = True
                self.opts['trainRHO'] = True
                self.opts['trainx0'] = True
                self.opts['weights']['udat'] = None
                self.opts['weights']['res'] = 1.0

                self.opts['earlystop_opts']['monitor'] = ['pdattest']
                self.opts['learning_rate_opts']['initial_learning_rate'] = 1e-4
                self.opts['num_init_train'] = 50000
            
            elif simtype == 'petonly':
                w = self.opts['patientweight']
                self.opts['trainD'] = True
                self.opts['trainRHO'] = True
                self.opts['weights']['udat'] = None
                self.opts['weights']['res'] = 1.0
                self.opts['weights']['seg1'] = None
                self.opts['weights']['seg2'] = None
                self.opts['earlystop_opts']['monitor'] = ['pdattest']
            
            elif simtype == 'segonly':
                w = self.opts['patientweight']
                self.opts['trainD'] = True
                self.opts['trainRHO'] = True
                self.opts['weights']['udat'] = None
                self.opts['weights']['res'] = 1.0
                self.opts['weights']['petmse'] = None
                self.opts['weights']['Areg'] = None
                self.opts['weights']['mreg'] = None
                self.opts['earlystop_opts']['monitor'] = ['pdattest']

            # quick test
            elif simtype == 'smalltest':
                self.opts['file_log'] = False
                self.opts['num_init_train'] = 100
                self.opts['lbfgs_opts']['maxfun'] = 20
                self.opts['print_res_every'] = 10
                self.opts['N'] = 64
                self.opts['Ntest'] = 64
                self.opts['Ndat'] = 64
                self.opts['Ndattest'] = 64
            
            elif simtype == 'smallnet':
                # testing with small network
                self.opts['nn_opts']['num_hidden_layers'] = 2
                self.opts['nn_opts']['num_neurons_per_layer'] = 16
                self.opts['geonn_opts']['depth'] = 2
                self.opts['geonn_opts']['width'] = 16
                
            
            elif simtype == 'noadam':
                self.opts['num_init_train'] = 0
            
            else:
                raise ValueError(f'simtype == {simtype} not recognized')
            
    def set_trainable(self):
        # given weight, set parameters to be trainable
        if self.opts['weights']['petmse'] is not None:
            self.opts['trainm'] = True
            self.opts['trainA'] = True
            self.opts['weights']['Areg'] = 1.0
            self.opts['weights']['mreg'] = 1.0
            
        
        if self.opts['weights']['seg1'] is not None:
            self.opts['trainth1'] = True
            self.opts['weights']['th1reg'] = 1.0
        
        if self.opts['weights']['seg2'] is not None:
            self.opts['trainth2'] = True
            self.opts['weights']['th2reg'] = 1.0
        
        if self.opts['weights']['adcmse'] is not None:
            self.opts['trainkadc'] = True
            self.opts['weights']['kadcreg'] = 1.0
    

if __name__ == "__main__":
    
    
    opts = Options()
    opts.parse_args(*sys.argv[1:])

    print (json.dumps(opts.opts, indent=2,sort_keys=True))
    


