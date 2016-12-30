
#from keras.models import Sequential, Graph
#from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, Adadelta
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras.objectives import categorical_crossentropy
from keras.layers import Input, Dense, Lambda, Dropout, Merge
from keras.models import Model
from keras import backend as K
from keras import objectives
import numpy as np
import tensorflow as tf
#from theano.compile.nanguardmode import NanGuardMode

def weighted_loss(base_loss,l):
    def loss_function(y_true, y_pred):
        return l*base_loss(y_true,y_pred)
    return loss_function
    
def add_dense_layer(model,dim,params,input_shape=None,act=LeakyReLU,alpha=0.1,dropout=True,batch_norm=True):
    if input_shape is not None:    
        model.add(Dense(dim,input_shape=input_shape,init='glorot_uniform'))
    else:
        model.add(Dense(dim,init='glorot_uniform'))
    model.add(act(alpha))
    #activation="relu"
    #model.add(act)
    if 'dropout' in params and params['dropout'] and dropout:
        model.add(Dropout(params['dropout_p']))
    if 'batch_norm' in params and params['batch_norm'] and batch_norm:
        model.add(BatchNormalization(dim))
        
def VAEenc(params,batch_size,original_dim,intermediate_dim,latent_dim):
    x = Input(batch_shape=(batch_size, original_dim))
    h = Dense(intermediate_dim, activation='relu',init='glorot_uniform')(x)
    #add_dense_layer(h,intermediate_dim,params)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    
    return x,z_mean,z_log_var
    
def VAEdec(params,in_z,x,intermediate_dim,original_dim):
    # we instantiate these layers separately so as to reuse them later
    decoder_h = Dense(intermediate_dim, activation='relu')(in_z)
    x_decoded_mean = Dense(original_dim, activation='relu',init='glorot_uniform')(decoder_h)
    x_decoded_log_std = Dense(original_dim, activation='relu',init='glorot_uniform')(decoder_h)
    #logpxz = 0.5* K.sum(x_decoded_log_std + K.square(x-x_decoded_mean)/K.exp(x_decoded_log_std),axis=-1)
    logpxz = 0.5* tf.reduce_sum(x_decoded_log_std + tf.square(x-x_decoded_mean)/tf.exp(x_decoded_log_std))
    #reconstr_loss = tf.reduce_sum(x * tf.log(1e-10 + x_reconstr_mean) + (1-x) * tf.log(1e-10 + 1 - x_reconstr_mean), 1)
    #decoder_h = Dense(intermediate_dim, activation='relu')
    #decoder_mean = Dense(original_dim, activation='sigmoid')
    #h_decoded = decoder_h(in_z)
    #x_decoded_mean = decoder_mean(h_decoded)
    #logpxz=1
    
    return x_decoded_mean,logpxz


def VAE(params,batch_size,original_dim,intermediate_dim,latent_dim):
    epsilon_std = 1.0
    
    print 'building enc...'
    x,z_mean,z_log_var = VAEenc(params,batch_size,original_dim,intermediate_dim,latent_dim)
    
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                  std=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    
    print 'building dec...'
    x_decoded_mean,logpxz = VAEdec(params,z,x,intermediate_dim,original_dim)
    
    def vae_loss(x, x_decoded_mean):#,logpxz
        #xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return logpxz + kl_loss#logpxz + 
        
    print 'compile...'
    vae = Model(x, x_decoded_mean)
    vae.compile(optimizer='rmsprop', loss=vae_loss)    
    
    return vae

#def VAEencS(in_x,params,batch_size,original_dim,intermediate_dim,latent_dim,s_dim):
#    
#    h = Dense(intermediate_dim, activation='relu',init='glorot_uniform')(in_x)
#    #add_dense_layer(h,intermediate_dim,params)
#    z_mean = Dense(latent_dim)(h)
#    z_log_var = Dense(latent_dim)(h)
#    s = Dense(s_dim)(h)
#    
#    return x,z_mean,z_log_var,s
    
#def VAEdecS(params,in_z,in_s,x_orig,intermediate_dim,original_dim):
#    # we instantiate these layers separately so as to reuse them later
#    inz_s = Merge([in_z, in_s], mode='concat', concat_axis=1)
#    decoder_h = Dense(intermediate_dim, activation='relu')(inz_s)
#    x_decoded_mean = Dense(original_dim, activation='relu',init='glorot_uniform')(decoder_h)
#    x_decoded_log_std = Dense(original_dim, activation='relu',init='glorot_uniform')(decoder_h)
#    logpxz = 0.5* tf.reduce_sum(x_decoded_log_std + tf.square(x_orig - x_decoded_mean)/tf.exp(x_decoded_log_std))
#    
#    return x_decoded_mean,logpxz    
    
def Adv(params,in_x,in_lbl,l_size):
    inx_lbl = Merge([in_x, in_lbl], mode='concat', concat_axis=1)  
    decoder_h = Dense(l_size, activation='relu')(inx_lbl)    
    decoder_h = Dense(l_size, activation='relu',init='glorot_uniform')(decoder_h)
    decoder_h = Dense(2, activation='softmax',init='glorot_uniform')(decoder_h)
    
def DistNet(params,batch_size,original_dim,intermediate_dim,latent_dim,s_dim,l_size):
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
    
    inz_s = Merge([in_z, in_s], mode='concat', concat_axis=1)
    decoder_h = Dense(intermediate_dim, activation='relu')(inz_s)
    x_decoded_mean = Dense(original_dim, activation='relu',init='glorot_uniform')(decoder_h)
    x_decoded_log_std = Dense(original_dim, activation='relu',init='glorot_uniform')(decoder_h)
    logpxz = 0.5* tf.reduce_sum(x_decoded_log_std + tf.square(in_x - x_decoded_mean)/tf.exp(x_decoded_log_std))
    
    VAEdecS = Model([in_z,in_s,in_x],[x_decoded_mean,logpxz])
    
    #Adv ########################################
    inx_lbl = Merge([in_x, in_lbl], mode='concat', concat_axis=1)  
    decoder_h = Dense(l_size, activation='relu')(inx_lbl)    
    decoder_h = Dense(l_size, activation='relu',init='glorot_uniform')(decoder_h)
    disc = Dense(1, activation='sigmoid',init='glorot_uniform')(decoder_h)
    
    Adv = Model([in_x,in_lbl],disc)
    ########################################
    
    
    print 'building encs...'
    x1 = Input(batch_shape=(batch_size, original_dim))
    x1t = Input(batch_shape=(batch_size, original_dim))
    x2 = Input(batch_shape=(batch_size, original_dim))
        
    x1,z1_mean,z1_log_var,s1 = VAEencS(x1)
    x1t,z1t_mean,z1t_log_var,s1t = VAEencS(x1t)
    x2,z2_mean,z2_log_var,s2 = VAEencS(x2)
    
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                  std=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z1 = Lambda(sampling, output_shape=(latent_dim,))([z1_mean, z1_log_var])
    zn=np.random.randn(latent_dim)
       
    print 'building dec...'
    x11,logpxz1 = VAEdecS(z1,s1,x1)
    x11t,logpxz1t = VAEdecS(z1,s1t,x1)
    x12,logpxz2 = VAEdecS(z1,s2,x1)
    xp2,logpxzp2 = VAEdecS(zn,s2,x1)
    
    def vae_loss1():
        #xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z1_log_var - K.square(z1_mean) - K.exp(z1_log_var), axis=-1)
        return logpxz1 + kl_loss 
    def vae_loss2():
        #xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z1t_log_var - K.square(z1t_mean) - K.exp(z1t_log_var), axis=-1)
        return logpxz1t + kl_loss
    def vae_loss3():
        #xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z1_log_var - K.square(z1_mean) - K.exp(z1_log_var), axis=-1)
        return logpxz1 + kl_loss
    print 'compile...'
    vae = Model(input=[x1,x1t,x2], output=[x11,x11t,x12,xp2])
    vae.compile(optimizer='rmsprop', loss=[vae_loss1,vae_loss2,vae_loss3])    
    
    return vae   