# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 11:01:09 2016

@author: algo
"""

from keras.layers import Input, Dense, Lambda, merge
from keras.models import Model
from keras import backend as K
import numpy as np

batch_size = 100
original_dim = 250
latent_dim = 12
intermediate_dim = 100
nb_epoch = 50
s_dim = 5
l_size=30
params={}
params['recweight'] = 0.5
params['swap1weight'] = 1
params['swap2weight'] = 0.01
params['klweightZ'] = 0.1

recweight = params['recweight'] if 'recweight' in params else 0.5
swap1weight = params['swap1weight'] if 'swap1weight' in params else 1
swap2weight = params['swap2weight'] if 'swap2weight' in params else 0.01
klweightZ = params['klweightZ'] if 'klweightZ' in params else 0.1 

epsilon_std = 1.0
in_x = Input(batch_shape=(batch_size, original_dim))
in_lbl = Input(batch_shape = (batch_size, 1))

#Enc ########################################
h = Dense(intermediate_dim, activation='relu',init='glorot_uniform')(in_x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
s = Dense(s_dim)(h)

VAEencS = Model(in_x, output=[z_mean,z_log_var,s])

#Dec ########################################
in_z = Input(batch_shape=(batch_size, latent_dim))
in_s = Input(batch_shape=(batch_size, s_dim))

inz_s = merge([in_z, in_s], mode='concat', concat_axis=1)
decoder_h = Dense(intermediate_dim, activation='relu')(inz_s)
x_decoded_mean = Dense(original_dim, activation='relu',init='glorot_uniform')(decoder_h)
x_decoded_log_std = Dense(original_dim, activation='relu',init='glorot_uniform')(decoder_h)
#logpxz = 0.5* tf.reduce_sum(x_decoded_log_std + tf.square(in_x - x_decoded_mean)/tf.exp(x_decoded_log_std))

VAEdecS = Model([in_z,in_s,in_x],[x_decoded_mean,x_decoded_log_std])#logpxz
########################################


print 'building encs...'
x1 = Input(batch_shape=(batch_size, original_dim))
x1t = Input(batch_shape=(batch_size, original_dim))
x2 = Input(batch_shape=(batch_size, original_dim))
zn = Input(batch_shape=(batch_size, latent_dim))
    
z1_mean,z1_log_var,s1 = VAEencS(x1)
z1t_mean,z1t_log_var,s1t = VAEencS(x1t)
z2_mean,z2_log_var,s2 = VAEencS(x2)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              std=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z1 = Lambda(sampling, output_shape=(latent_dim,))([z1_mean, z1_log_var])
#zn=np.random.randn(batch_size,latent_dim)
#zn=Lambda(lambda x: x)(K.random_normal(shape=(batch_size, latent_dim), mean=0.,std=1))

print 'building dec...'
x11,x11_log_std = VAEdecS([z1,s1,x1])
x11t,x11t_log_std = VAEdecS([z1,s1t,x1])
x12,x12_log_std = VAEdecS([z1,s2,x1])
xp2,xp2_log_std = VAEdecS([zn,s2,x1])


DistNet = Model([x1,x1t,x2,zn], [x11,x11t,x12,xp2])
  

