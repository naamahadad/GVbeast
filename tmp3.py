import numpy as np
#import matplotlib.pyplot as plt
#from scipy.stats import norm
#
#from keras.layers import Input, Dense, Lambda
#from keras.models import Model
#from keras import backend as K
#from keras import objectives
#from keras.datasets import mnist
from time import strftime,localtime
import sys
import os
import KerasModels
import yaml
import dataFactory_pyprep
import keras
import DistNet
from keras.models import load_model

home = os.path.expanduser('~')

params = {}
debug = True

params['config_file'] = sys.argv[1] if len(sys.argv)>1 else 'config_adam_epoch5000.yaml'
params['data'] = [home +'/FinData/prices_debug.hdf']

#with open('yamls/' + params['config_file'],'r') as f:
#    params.update(yaml.load(f))

params['res_path'] = home + '/results_nn/VAE'
params['years_dict'] = {'train_top' : 2012, # 2009
                      'test_bottom' : 2013, # 2010
                      'test_top' : 2015} # 2012

params['recweight'] = 1
params['swap1weight'] = 1
params['swap2weight'] = 1#0.01
params['klweightZ'] = 0.1#0.1

batch_size = 100
nb_epoch = 900
params['batch_size'] = batch_size
params['nb_epoch'] = nb_epoch

params['original_dim'] = 5
params['latent_dim'] = 3
params['intermediate_dim'] = 5
params['s_dim'] = 2
params['l_size'] = 5

#vae = KerasModels.VAE(params,batch_size,original_dim,intermediate_dim,latent_dim)
net = DistNet.DistNet(params,'','',False)
#net.VAEencS.load_weights(params['res_path'] +'/weights/231216_102327_config_adam_epoch5000_l3_0p1_on.yaml_encS_weights.h5')
#net.DistNet.load_weights(params['res_path'] +'/weights/231216_102327_config_adam_epoch5000.yaml_full_weights.h5')

#net.VAEencS.load_weights(params['res_path'] +'/weights/231216_094509_config_adam_epoch5000_l1_2.yamlencS_weights.h5')
#net.DistNet.load_weights(params['res_path'] +'/weights/231216_094509_config_adam_epoch5000_l1_l2.yamlfull_weights.h5')

#net.VAEencS.load_weights(params['res_path'] +'/weights/231216_105710_config_adam_epoch5000_l1_l2_adam.yaml_encS_weights.h5')
#net.DistNet.load_weights(params['res_path'] +'/weights/231216_105710_config_adam_epoch5000.yaml_full_weights.h5')

#net.VAEencS.load_weights(params['res_path'] +'/weights/231216_143933_config_adam_epoch5000_all_100k.yaml_encS_weights.h5')
#net.DistNet.load_weights(params['res_path'] +'/weights/231216_143933_config_adam_epoch5000.yaml_full_weights.h5')

#net.VAEencS.load_weights(params['res_path'] +'/weights/231216_145952_config_adam_epoch5000_alll3off.yaml_encS_weights.h5')
#net.DistNet.load_weights(params['res_path'] +'/weights/231216_145952_config_adam_epoch5000.yaml_full_weights.h5')

#net.VAEencS.load_weights(params['res_path'] +'/weights/231216_202827_config_adam_epoch5000_3off_40p1.yaml_encS_weights.h5')
#net.DistNet.load_weights(params['res_path'] +'/weights/231216_202827_config_adam_epoch5000.yaml_full_weights.h5')

#net.VAEencS.load_weights(params['res_path'] +'/weights/231216_204356_config_adam_epoch5000_l1_l2.yaml_encS_weights.h5')
#net.DistNet.load_weights(params['res_path'] +'/weights/231216_204356_config_adam_epoch5000.yaml_full_weights.h5')

#net.VAEencS.load_weights(params['res_path'] +'/weights/231216_205121_config_adam_epoch5000_l3off_l40p01.yaml_encS_weights.h5')
#net.DistNet.load_weights(params['res_path'] +'/weights/231216_205121_config_adam_epoch5000.yaml_full_weights.h5')

#net.VAEencS.load_weights(params['res_path'] +'/weights/231216_210307_config_adam_epoch5000_l3_4off_nolayers.yaml_encS_weights.h5')
#net.DistNet.load_weights(params['res_path'] +'/weights/231216_210307_config_adam_epoch5000.yaml_full_weights.h5')

#net.VAEencS.load_weights(params['res_path'] +'/weights/231216_212015_config_adam_epoch5000_l3off_l40p05_nolayers.yaml_encS_weights.h5')
#net.DistNet.load_weights(params['res_path'] +'/weights/231216_212015_config_adam_epoch5000.yaml_full_weights.h5')

#net.VAEencS.load_weights(params['res_path'] +'/weights/231216_215113_config_adam_epoch5000_paper_mb20.yaml_encS_weights.h5')
#net.DistNet.load_weights(params['res_path'] +'/weights/231216_215113_config_adam_epoch5000.yaml_full_weights.h5')

net.VAEencS.load_weights(params['res_path'] +'/weights/241216_135945_config_adam_epoch5000.yaml_encS_weights.h5')
net.DistNet.load_weights(params['res_path'] +'/weights/241216_135945_config_adam_epoch5000.yaml_full_weights.h5')

num_ids=10
nSamples = 100
sig = 0#.005
s_rand = np.random.randint(-num_ids/2,num_ids/2+1,(nSamples,params['s_dim'] ))
z_rand = np.random.randn(nSamples,params['latent_dim'])
x1_z = np.repeat(z_rand,1,axis=1) + sig*np.random.rand(z_rand.shape[0],params['latent_dim']*1)
x1_s = np.repeat(s_rand,1,axis=1) + sig*np.random.rand(s_rand.shape[0],params['s_dim']*1)
x1_z[:,:1] = x1_z[:,:1] + x1_s[:,:1]
x1_z[:,1:2] = x1_z[:,1:2] + x1_s[:,1:2]
x1 = np.concatenate((x1_z,x1_s),axis=1)

z_rand = np.random.randn(nSamples,params['latent_dim'])
x2_z=np.repeat(z_rand,1,axis=1) + sig*np.random.rand(z_rand.shape[0],params['latent_dim']*1)
x2_z[:,:1] = x2_z[:,:1] + x1_s[:,:1]
x2_z[:,1:2] = x2_z[:,1:2] + x1_s[:,1:2]
x2 = np.concatenate((x2_z,x1_s),axis=1)

z_rand = np.random.randn(nSamples,params['latent_dim'])
s_add = np.random.randint(1,num_ids,(nSamples,params['s_dim'] ))
s_rand = np.mod(s_rand+num_ids/2+s_add,num_ids)-num_ids/2
x3_z = np.repeat(z_rand,1,axis=1) + sig*np.random.rand(z_rand.shape[0],params['latent_dim']*1)
x3_s = np.repeat(s_rand,1,axis=1) + sig*np.random.rand(s_rand.shape[0],params['s_dim']*1)
x3_z[:,:1] = x3_z[:,:1] + x3_s[:,:1]
x3_z[:,1:2] = x3_z[:,1:2] + x3_s[:,1:2]
x3 = np.concatenate((x3_z,x3_s),axis=1)
x_train = np.concatenate((x1,x2,x3),axis=1)
id_train = (np.sum(x3,axis=1)>0) + 1 -1


X1 = x_train[:, :x_train.shape[1]/3]
X1t = x_train[:, (x_train.shape[1]/3):(2*x_train.shape[1]/3)]
X2 = x_train[:, (2*x_train.shape[1]/3):x_train.shape[1]]
lbls = id_train

zn=np.random.randn(100,params['latent_dim'])

z1_m,z1_std,s1 = net.VAEencS.predict(X1,batch_size=params['batch_size'])
X11,X11t,X12,Xp2,adv1,adv2 = net.DistNet.predict([X1,X1t,X2,zn,lbls],batch_size=params['batch_size'])
print 'loss1: '+ str(np.sum(np.square(X11-x1)))+' loss2: '+ str(np.sum(np.square(X11t-x1)))
np.average(np.square(X11-x1),axis=0)

vv=net.DistNet.get_layer(index=2)