import tensorflow as tf
# Define model architecture
class PINN(tf.keras.Model):
    """ Set basic architecture of the PINN model."""

    def __init__(self,
            output_dim=1,
            input_dim=1,
            num_hidden_layers=3, 
            num_neurons_per_layer=100,
            activation='tanh',
            kernel_initializer='glorot_normal',
            output_transform = lambda x,u:u,
            param = None,
            resnet = False,
            regularizer = None,
            userff = False,
            userbf = False,
            **kwargs):
        super().__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.num_neurons_per_layer = num_neurons_per_layer
        self.resnet = resnet

        # phyiscal parameters in the model
        self.param = param

        if activation == 'sin':
            activation_fcn = tf.sin
            kernel_initializer = SineInit()
        else:
            activation_fcn = tf.keras.activations.get(activation)
            

        # Define NN architecture
        self.hidden = [tf.keras.layers.Dense(num_neurons_per_layer,
                             activation=activation_fcn,
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer= regularizer)
                           for _ in range(self.num_hidden_layers)]
        
        

        if userff == True:
            rbf =  tf.keras.layers.experimental.RandomFourierFeatures(
                output_dim=num_neurons_per_layer,
                scale=1.,
                trainable = True,
                kernel_initializer='gaussian')
            self.hidden =   [rbf] + self.hidden
        
        if userbf == True:
            rbf =  RBFLayer(num_neurons_per_layer, betas=1.)
            self.hidden =   [rbf] + self.hidden


        self.out = tf.keras.layers.Dense(output_dim)

        
        self.output_transform = output_transform
        
        self.build(input_shape=(None,input_dim))

        if activation == 'sin':
            # first layer is re-initilize, 
            first_layer = self.hidden[0]
            weights, biases = first_layer.get_weights()
            initializer = SineFirstInit()
            new_weights = initializer(shape=weights.shape, dtype=weights.dtype)
            new_biases = initializer(shape=biases.shape, dtype=biases.dtype)

            first_layer.set_weights([new_weights, new_biases])
    
    @tf.function
    def call(self, X):
        """Forward-pass through neural network."""
        Z = X

        if self.resnet != True:
            for i in range(len(self.hidden)):
                Z = self.hidden[i](Z)
        else:
            # resnet implementation
            Z = self.hidden[0](Z)
            for i in range(1,len(self.hidden)-1,2):
                Z = self.hidden[i+1](self.hidden[i](Z))+Z
            if i + 2 < len(self.hidden):
                Z = self.hidden[i+2](Z)
            
        Z = self.out(Z)
        Z = self.output_transform(X,Z)
        return Z
    
    def lastlayer(self,X):
        '''output last layer output, weights, bias'''
        if self.resnet == True:
            sys.exit('intermediate output not for resnet')
        Z = X
        Zout = []
        for i in range(len(self.hidden)):
                Z = self.hidden[i](Z)
                Zout.append(Z)
        return Zout, self.out.weights[0], self.out.weights[1]
    
    def set_trainable_layer(self, arg):
        '''Set trainable property of each layer
        '''
        if arg == False:
            print("do not train NN")
            # self.model.trainable = False
            for l in self.layers:
                l.trainable = False
        
        # set layer to be trainable
        if isinstance(arg,int):
            nlayer = arg
            k = 0
            print(f"do not train nn layer <= {nlayer} ")
            # self.trainable = False
            for l in self.layers:
                if k > nlayer:
                    l.trainable = True
                else:
                    l.trainable = False
                k += 1



# SIREN paper
# https://github.com/vsitzmann/siren/blob/master/modules.py
class SineInit(tf.keras.initializers.Initializer):

    def __call__(self, shape, **kwargs):
        return tf.random.uniform(shape, -tf.sqrt(6.0/shape[-1]), tf.sqrt(6.0/shape[-1]))

class SineFirstInit(tf.keras.initializers.Initializer):

    def __call__(self, shape, **kwargs):
        return tf.random.uniform(shape, -1/shape[-1], 1/shape[-1])
