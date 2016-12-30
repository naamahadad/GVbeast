
import numpy 
#import scipy.io
#import h5py
import pandas as pd
#import cPickle as pickle
#from keras.utils import np_utils
#import pdb
from joblib import Parallel, delayed
from tempfile import TemporaryFile

#from math import floor

class dataFactory_pyprep(object):
    
    def __init__(self, years_dict,data_path,data_tags,exchange_ids=None,s_mode=0,load_ready_data=False):
        
        dataroot = '/media/data2/naamahadad/PyData/smode1'

        if load_ready_data:
            self.train_data=pd.read_hdf(dataroot+'_train_data.hdf','table')
            self.test_data=pd.read_hdf(dataroot+'_test_data.hdf','table')
            self.train_lbls = numpy.load(dataroot+'_train_lbls.npy')
            self.test_lbls = numpy.load(dataroot+'_test_lbls.npy')
            self.data_tags = numpy.load(dataroot+'_data_tags.npy')
            return

        #full_data = pd.concat(pd.read_hdf(cur_path,'table') for cur_path in data_path)
        full_data = pd.read_hdf(data_path,'table')
             
        #full_data.set_index('year',drop=True,inplace=True)
        full_data.sector = pd.to_numeric(full_data.sector, errors='coerce')
        print 'sector nan',numpy.sum(numpy.isnan(full_data.sector.values))
        full_data.sector.fillna(0, inplace=True)
        full_data.sector = numpy.floor(full_data.sector/100)
        
        if s_mode==1:
            full_data.insert(full_data.shape[1],'groups_val',full_data['year'])
        elif s_mode==2:
            full_data.insert(full_data.shape[1],'groups_val',full_data['year']*100+full_data['sector'])
            
        train_data = full_data[full_data.year<=years_dict['train_top']]
        test_data = full_data[full_data.year>=years_dict['test_bottom']]
        test_data = test_data[test_data.year<=years_dict['test_top']]
        
        
#        if train_data.shape[0] != 0:
#            train_labels = [numpy.where(train_data['class_label'] == 1)[0],
#                            numpy.where(train_data['class_label'] == 2)[0]]          
#        
#            # level datas
#            num_samples = [len(train_labels[0]),len(train_labels[1])]
#            sorted_numsamples = numpy.argsort(num_samples)
#            new_labels = numpy.random.choice(train_labels[sorted_numsamples[1]],size=num_samples[sorted_numsamples[0]],replace=False)            
#            res_train_labels = numpy.concatenate((numpy.asarray(train_labels[sorted_numsamples[0]]),new_labels),axis=0)
#            
#            train_data = train_data.iloc[res_train_labels]
#            train_data.reset_index(inplace=True,drop=False)
#        
#        test_data.reset_index(inplace=True,drop=False)
            
        self.train_data = train_data
        self.test_data = test_data
        self.data_tags = data_tags
        self.s_mode = s_mode
        if s_mode>0:
            self.train_data = self.build_s_data(train_data,s_mode=s_mode)
            self.test_data = self.build_s_data(test_data,s_mode=s_mode)
            aa = [col for col in self.train_data.columns if 'day' in col]
            self.data_tags = aa
            
            
            self.train_lbls = (self.train_data.groups_val_2.values.astype(numpy.float32)-195000)
            self.test_lbls = (self.test_data.groups_val_2.values.astype(numpy.float32)-195000)

        print 'done! saving to',dataroot
        self.train_data.to_hdf(dataroot+'_train_data.hdf','table')
        self.test_data.to_hdf(dataroot+'_test_data.hdf','table')
        numpy.save(dataroot+'_train_lbls', self.train_lbls)
        numpy.save(dataroot+'_test_lbls', self.test_lbls)
        numpy.save(dataroot+'_data_tags', self.data_tags)
        
    def build_s_data(self,data,s_mode=1):
        global ind
        ind=0
        def build_s_year(df):
		
            global ind
            print ind
            couples_perm = numpy.random.permutation(range(numpy.floor(df.shape[0]/2).astype(numpy.int)*2))
            couples_perm = numpy.reshape(couples_perm,(couples_perm.shape[0]/2,2))
            firstDF = df.iloc[couples_perm[:,0],:]
            secondDF = df.iloc[couples_perm[:,1],:]
            secondDF.rename(columns=lambda x: x+'_1', inplace=True)
            newDf = pd.concat((firstDF,secondDF),axis=1)

            firstDF = df.iloc[couples_perm[:,1],:]
            secondDF = df.iloc[couples_perm[:,0],:]
            secondDF.rename(columns=lambda x: x+'_1', inplace=True)
            newDf1 = pd.concat((newDf,pd.concat((firstDF,secondDF),axis=1)),axis=0)
            ind = ind+1
            return newDf1
            
        def applyParallel(dfGrouped, func):
            retLst = Parallel(n_jobs=num_workers)(delayed(func)(ind,group) for ind,(name, group) in enumerate(dfGrouped))
            #retLst = func(ind,group) for ind,(name, group) in enumerate(dfGrouped)
            #retLst = extract_data_stock(ind,group) for ind,(name, group) in enumerate(dfGrouped)
            return pd.concat(retLst)
            
        num_workers = 1
        
        #count_years = data['year'].value_counts()
        #newTrain = pd.DataFrame(numpy.sum(numpy.floor(count_years.values/2)),self.train_data.shape[1])
        #newTrain = applyParallel(self.train_data.groupby('year'),build_s_year)
        #if s_mode==1:
        #    newTrain = data.groupby('year').apply(build_s_year)
        #    newTrain.groups_val = newTrain['year']
        #elif s_mode==2:
        #    newTrain = data.groupby(['year','sector']).apply(build_s_year)
        #    newTrain.groups_val = newTrain['year']*100+newTrain['sector']
	#newTrain = applyParallel(data.groupby('groups_val'),build_s_year)
        newTrain = data.groupby('groups_val').apply(build_s_year)
            
        #years = newTrain['year'].unique()
        #newTrain.reset_index(drop=True,inplace=True)
        #newTrain = newTrain.groupby('groups_val')
        groups_val = newTrain['groups_val'].unique()
        #finalTrain = pd.DataFrame()
        for group in groups_val:
            thisYear = newTrain[newTrain.groups_val==group]
            thisYear.reset_index(drop=True,inplace=True)
            diffYears = newTrain[newTrain.groups_val!=group]#self.train_data[self.train_data.year!=year]
            thirdPerm = numpy.random.permutation(range(diffYears.shape[0]))
            thirdPerm = thirdPerm[:thisYear.shape[0]]
            
            secondDF1 = diffYears.iloc[thirdPerm,:diffYears.shape[1]/2]
            secondDF1.rename(columns=lambda x: x+'_2', inplace=True)
            secondDF1.reset_index(drop=True,inplace=True)
            newDf = pd.concat((thisYear,secondDF1),axis=1)
            finalTrain = newDf if group==groups_val[0] else pd.concat((finalTrain,newDf),axis=0)
        
        return finalTrain
    def get_train_data(self):
        #y = np_utils.to_categorical(self.train_data['class_label'].values -1)
        #X = self.train_data[self.data_tags].values.astype(numpy.float32)
    
        return self.train_data[self.data_tags].values.astype(numpy.float32),self.train_lbls

    def get_test_data(self):          
        return self.test_data,self.test_lbls

				
