'''
nn for geometry
'''
from DataSet import DataSet
import tensorflow as tf
import os
import sys

class Geonn(tf.keras.Model):
    """ nn for geometry"""

    def __init__(self,
            input_dim=3,
            num_hidden_layers=2, 
            num_neurons_per_layer=64,
            **kwargs):
        super().__init__( **kwargs)
        self.input_dim = input_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_per_layer = num_neurons_per_layer
        
        # Input layer
        self.input_layer = tf.keras.layers.Dense(self.num_neurons_per_layer, activation='tanh', input_shape=(self.input_dim,))

        # Hidden layers
        self.hidden_layers = [tf.keras.layers.Dense(self.num_neurons_per_layer, activation='tanh') for _ in range(self.num_hidden_layers - 1)]

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
        batch_size = X.shape[0]
        history = self.fit(X, {'Pwm': Pwmq,  'Pgm': Pgmq, 'phi': phiq}, epochs=epochs, batch_size=batch_size)
        return history

    # def save_checkpoint(self):
    #     self.checkpoint_manager.save()

    # def load_checkpoint(self, checkpoint_path=None):
    #     if checkpoint_path:
    #         self.checkpoint.restore(checkpoint_path)
    #     else:
    #         self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)    

if __name__ == '__main__':
    
    fname = sys.argv[1]
    N = 50000
    epochs = 10000

    dataset = DataSet(fname)
    dataset.downsample(N)
    geonn = Geonn()
    geonn.train(dataset.xr[:,1:], dataset.Pwmq, dataset.Pgmq, dataset.phiq, epochs)
    geonn.save_checkpoint()

    