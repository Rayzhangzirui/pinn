import sys
from config import *
from scipy.io import loadmat
import numpy as np

# load .mat data
class DataSet:
    def __init__(self, matfile) -> None:
        
        assert os.path.exists(matfile), f'{matfile} not exist'
        
        _ , ext = os.path.splitext(matfile)
        assert ext == '.mat', 'not reading mat file'
        
        matdat = loadmat(matfile,mat_dtype=True)

        for key, value in matdat.items():
            
            if key.startswith("__"):
                # skip meta data
                continue
            if isinstance(value,np.ndarray):
                if value.dtype.kind in {'f','i','u'}:
                    # convert to float or double
                    value = value.astype(DTYPE)
                if value.size == 1:
                    # if singleton, get number
                    value = value.item()
                
                setattr(self, key, value)

        self.dim = self.xr.shape[1]
        self.xdim = self.xr.shape[1]-1
    
    def print(self):
        '''print data set'''
        attr = [a for a in dir(self) if not a.startswith("__") and not callable(getattr(self,a))]
        for a in attr:
            x = getattr(self, a)
            print(f"{a} {x}")
            if isinstance(x, np.ndarray):
                print(f"{x.shape} {x.dtype}")
    
    def downsample(self,n):
        ''' downsample data size
        '''
        # get variable name in .mat, remove meta info
        attr = [a for a in dir(self) if not a.startswith("__") and not callable(getattr(self,a))]
        for a in attr:
            x = getattr(self, a)
            # only work on variables with more than one rows
            if isinstance(x,np.ndarray) and x.shape[0]>n:
                print(f'downsample {a} from {x.shape[0]} to {n} ')
                x = x[:n,:]
                setattr(self, a, x)

    def subsample(self, idx):
        ''' subsample data set
        '''
        attr = [a for a in dir(self) if not a.startswith("__") and not callable(getattr(self,a))]
        for a in attr:
            x = getattr(self, a)
            # only work on variables with more than one rows
            if isinstance(x,np.ndarray) and x.shape[0]>len(idx):
                print(f'subsample {a} from {x.shape[0]} to {len(idx)} ')
                x = x[idx,:]
                setattr(self, a, x)
    
    


            



if __name__ == "__main__":
    filename  = sys.argv[1]
    dataset = DataSet(filename)
    dataset.print()