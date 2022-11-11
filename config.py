import os
import tensorflow as tf

# Set data type
DTYPE='float64'
tf.keras.backend.set_floatx(DTYPE)


hostname = os.uname()[1].lower()

if 'gru' in hostname:
    DATADIR='/mnt/data/rzhang/pinndata'
elif 'hpc3' in hostname:
    DATADIR='~/pinndata'
elif 'poison' in hostname:
    DATADIR='/home/ziruz16/models'
else:
    DATADIR='./'



