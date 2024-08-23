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

        self.arraynames = []
        for key, value in matdat.items():
            
            if key.startswith("__"):
                # skip meta data
                continue
            if isinstance(value,np.ndarray):
                if value.dtype.kind in {'f','i','u'}:
                    # convert to float or double
                    value = value.astype(DTYPE)
                
                if value.shape[0]> 1:
                    self.arraynames.append(key)

                if value.size == 1:
                    # if singleton, get number
                    value = value.item()
                
                setattr(self, key, value)

        self.dim = self.X_res.shape[1]
        self.xdim = self.X_res.shape[1]-1
        
        # collection of attributes not callable
        self.attr = [a for a in dir(self) if not a.startswith("__") and not callable(getattr(self,a))]
        

    def print(self, attr=None):
        '''print data set
        '''
        # if attr is not None, print attr
        
        attr_to_print = self.attr if attr is None else attr
        # if attr is None:
        for a in attr_to_print:
            x = getattr(self, a)
            print(f"{a} {x}")
            if isinstance(x, np.ndarray):
                print(f"{x.shape} {x.dtype}")
    
    def downsample(self,n, names=None):
        ''' downsample data size
        '''
        names = self.arraynames if names is None else names
        # get variable name in .mat, remove meta info
        for a in names:
            # check if attribute exist
            if not hasattr(self, a):
                continue
            x = getattr(self, a)
            # only work on variables with more than one rows
            print(f'downsample {a} from {x.shape[0]} to {n} ')
            x = x[:n,:]
            setattr(self, a, x)

    def subsample(self, idx, names):
        ''' subsample data set
        '''
        names = self.arraynames if names is None else names
        for a in names:
            x = getattr(self, a)
            # only work on variables with more than one rows
            print(f'subsample {a} from {x.shape[0]} to {len(idx)} ')
            x = x[idx,:]
            setattr(self, a, x)
    
    
    def shuffle(self):
        ''' permute data set
        '''
        idx = np.random.permutation(self.X_res.shape[0])
        for a in self.arraynames:
            x = getattr(self, a)
            x = x[idx,:]
            setattr(self, a, x)

    


if __name__ == "__main__":
    # read mat file and print dataset
    filename  = sys.argv[1]
    vars2print = sys.argv[2:] if len(sys.argv) > 2 else None

    dataset = DataSet(filename)
    dataset.print(vars2print)