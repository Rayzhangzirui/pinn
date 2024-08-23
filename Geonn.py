'''
nn for geometry
'''
from DataSet import DataSet
import tensorflow as tf
import os
import sys
from options import Options
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger


class Geonn(tf.keras.Model):
    """ nn for geometry"""

    def __init__(self,
            input_dim=3,
            depth=2, 
            width=64,
            model_dir='geomodel',
            **kwargs):
        super().__init__( **kwargs)
        self.input_dim = input_dim
        self.depth = depth
        self.width = width
        self.model_dir = model_dir # place to save training info and checkpoint
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Input layer
        self.input_layer = tf.keras.layers.Dense(self.width, activation='tanh', input_shape=(self.input_dim,))

        # Hidden layers
        self.hidden_layers = [tf.keras.layers.Dense(self.width, activation='tanh') for _ in range(self.depth - 1)]

        # Output layers
        self.output_Pwm = tf.keras.layers.Dense(1, activation='sigmoid')
        self.output_Pgm = tf.keras.layers.Dense(1, activation='sigmoid')
        self.output_phi = tf.keras.layers.Dense(1, activation='sigmoid')

        self.build(input_shape=(None,input_dim))


    def call(self, inputs):
        x = self.input_layer(inputs)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)

        return {'Pwm':self.output_Pwm(x),'Pgm':self.output_Pgm(x),'phi':self.output_phi(x)}

    def train(self, X, Pwmq, Pgmq, phiq, epochs=10000):
        self.compile(optimizer='adam',loss={'Pwm': 'mse', 'Pgm': 'mse', 'phi': 'mse'})
        # do not use mini-batch
        # split_ratio = 0.8
        # idxsplit = int(X.shape[0]*split_ratio)
        # tfdataset = tf.data.Dataset.from_tensor_slices({'xdat':X, 'Pwm':Pwmq, 'Pgm':Pgmq, 'phi':phiq})
        
        # train = tfdataset.take(idxsplit)
        # test = tfdataset.skip(idxsplit)
        
        
        early_stopping_val = EarlyStopping(monitor='val_loss', patience=1000)
        early_stopping_loss = EarlyStopping(monitor='loss', patience=1000)
        csv_logger = CSVLogger(os.path.join(self.model_dir,'training.txt' ))

        
        batch_size = X.shape[0]
        history = self.fit(X, {'Pwm': Pwmq,  'Pgm': Pgmq, 'phi': phiq}, 
                            validation_split=0.2,
                           epochs=epochs, batch_size=batch_size,callbacks=[early_stopping_val, early_stopping_loss, csv_logger],
                           verbose=1)
        return history

    def setup_ckpt(self, ckptdir):
        self.checkpoint = tf.train.Checkpoint(self)
        self.manager = tf.train.CheckpointManager(self.checkpoint, directory=os.path.join(self.model_dir, ckptdir), max_to_keep=4)

    def save_checkpoint(self):
        # make director
        self.manager.save()

    def load_checkpoint(self):
        self.checkpoint.restore(self.manager.latest_checkpoint)
        

if __name__ == '__main__':
    
    args = sys.argv[1:]
    optobj = Options()
    optobj.parse_args(*sys.argv[1:])

    dataset = DataSet(optobj.opts['inv_dat_file'])

    dataset.downsample(optobj.opts['Ndat'])
    geonn = Geonn(input_dim=dataset.xdim, **(optobj.opts['geonn_opts']), model_dir = optobj.opts['model_dir'])
    geonn.setup_ckpt('geockpt')
    history = geonn.train(dataset.X_dat[:,1:], dataset.Pwmdat, dataset.Pgmdat, dataset.phi_dat, epochs=optobj.opts['num_init_train'])
    geonn.save_checkpoint()


    f = open("commands.txt", "a")
    f.write(' '.join(sys.argv))
    f.write('\n')
    f.write('finish\n')
    f.close()

    