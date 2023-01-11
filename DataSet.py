import sys
from config import *
from scipy.io import loadmat
import numpy as np


class DataSet:
    def __init__(self, opts) -> None:
        self.opts = opts
        
        
        inv_dat_file = opts['inv_dat_file']

        assert os.path.exists(inv_dat_file), f'{inv_dat_file} not exist'
        
        _ , ext = os.path.splitext(inv_dat_file)
        
        assert ext == '.mat', 'not reading mat file'
        

        matdat = loadmat(inv_dat_file)

        for key, value in matdat.items():
            if key.startswith("__"):
                continue
            if isinstance(value,np.ndarray):
                if value.dtype.kind == 'f':
                    # convert to float or double
                    value = value.astype(DTYPE)
                if value.size == 1:
                    # if singleton, get number
                    value = value.item()
                
                setattr(self, key, value)
        if self.opts.get('N') is not None:
            self.downsample(self.opts.get('N'))
        self.dim = self.xr.shape[1]
        self.xdim = self.xr.shape[1]-1
    
    def print(self):
        attr = [a for a in dir(self) if not a.startswith("__") and not callable(getattr(self,a))]
        for a in attr:
            x = getattr(self, a)
            print(f"{a} {x}")
    
    def downsample(self,n):
        attr = [a for a in dir(self) if not a.startswith("__") and not callable(getattr(self,a))]
        for a in attr:
            x = getattr(self, a)
            if isinstance(x,np.ndarray) and x.shape[0]>n:
                print(f'downsample {a} from {x.shape[0]} to {n} ')
                x = x[:n,:]
                setattr(self, a, x)
            



if __name__ == "__main__":
    filename  = sys.argv[1]
    opts = {}
    opts['inv_dat_file'] = filename
    dataset = DataSet(opts)
    dataset.print()