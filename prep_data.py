
import numpy
import pandas as pd
from collections import OrderedDict
import sys
import scipy
import os
import cPickle as pickle
import time
from joblib import Parallel, delayed
    
def extract_data_stock(ind,df,stepsize=250):
    print ind
     
    if df.shape[0] < stepsize:
        return pd.DataFrame()
        
    df = df.iloc[:stepsize,:]
    #df[df.RET == 'C'].RET = 0
    #df[df[['RET']]=='C'] = 0
    mask = (df['RET']=='C')
    df.RET = df.RET.where(~mask, other=0)
    #cur_year = applyParallel(df.groupby(df['date'].map(lambda x: x.year)),get_year_data)
    cur_rets = (df['RET'].values).astype(numpy.float16) # convert to flpat32 to save memory
    cur_prices_wo_adj = (df['PRC'].values).astype(numpy.float32)#need to add price col

    #if cur_rets.size < min_size:
    #    return pd.DataFrame()
    #first_date = df.first().date
    cur_year = df.iloc[0,:].date.year
    print cur_year 
    
    sw_rets = cur_rets
    ## go in a sliding-window manner on all prices, calculate a matrix sized (#samples,num_days)
    #sw_rets = numpy.vstack(cur_rets[i:i+stepsize][None,:] for i in range(cur_rets.size - stepsize))    
    
    ## convert prices to returns
    #sw_rets = numpy.roll(sw_prices,-1,axis=1) / sw_prices - 1 # roll (-1) means roll left
    #sw_rets = sw_rets[:,:-1] # don't take last sample due to roll
    if (sw_rets>0.99).any() or (sw_rets==-66).any() or (sw_rets==-77).any() or (sw_rets<=-88).any()  or (sw_rets<=-99).any():
        return pd.DataFrame()
    
    ## convert to cum rets
    #for col_ind in range(1,sw_rets.shape[1]):
    #    sw_rets[:,col_ind] = (1+sw_rets[:,col_ind-1])*(1+sw_rets[:,col_ind]) - 1

    ## go in a sliding-window manner on volumes, calculate a matrix sized (#samples,num_vols)
    #orig_vols = (df['VOL'].values)#.astype(numpy.int32) # convert to flpat32 to save memory
    #cur_vols = numpy.vstack(orig_vols[(i-num_vols):i][None,:] for i in range(stepsize, orig_vols.size))
    #vols2avg = numpy.vstack(orig_vols[(i-stepsize+1):i][None,:] for i in range(stepsize, orig_vols.size))
    #cur_vols = cur_vols/vols2avg.mean(axis=1)[:,None] -1
    
    orig_vols = (df['VOL'].values)
    
    #calculate avg,last and next day volume in dollars
    vols_dollars = orig_vols*cur_prices_wo_adj
    #last_vol_dollars=vols_dollars[stepsize-1:-1].tolist()
    #cur_vol_dollars=vols_dollars[stepsize:].tolist()
    
    #days2avg=5  
    #vols2avg = numpy.vstack(vols_dollars[(i-days2avg):i][None,:] for i in range(stepsize, orig_vols.size))
    avgvol = numpy.mean(vols_dollars)
    
    #avg,last and next day price in dollars
    last_price=cur_prices_wo_adj[-1]
    #cur_price=cur_prices_wo_adj[stepsize:].tolist()
    #prices2avg = numpy.vstack(cur_prices_wo_adj[(i-days2avg):i][None,:] for i in range(stepsize, orig_vols.size))
    avgprice = numpy.mean(cur_prices_wo_adj)
    
    ## calculate next-day-returns in order to calculate classes later on
    #returns = numpy.roll(cur_rets,-1)
    #returns = returns[:-1]
    #cur_nextdayrets = returns[stepsize-1::].tolist()

    # calculate the last taken day (out of num_days) for each num_days sized series    
    # don't take last day because last day is used for calculating a label only, and shouldn't be in any series
    #cur_nextdates = cur_dates[stepsize:]
        
    # convert to pandas dataframe
    days_cols = ['day'+str(i) for i in range(0,stepsize)]
    cur_symbol = df['TICKER'].values[0]
    cur_permno = df['PERMCO'].values[0]
    cur_mv = df['PRC'].values[0]*df['SHROUT'].values[0]
    #cur_vol = df['PRC'].values[0]*df['VOL'].values[0]
    cur_sector = df['SICCD'].values[0]#numpy.floor(df['SICCD'].values[0]/100)
    #days_cols_vol = ['vol_day'+str(i) for i in range(stepsize-num_vols,stepsize)]

    sw_rets = numpy.expand_dims(sw_rets, axis=0)
    
    
    df = pd.DataFrame(data=sw_rets,columns=days_cols)
    df.insert(0,'symbol',cur_symbol)
    df.insert(1,'permno',cur_permno)
    df.insert(2,'year',cur_year)
    df.insert(3,'avg_volume',avgvol)
    df.insert(4,'last_price',last_price)
    df.insert(5,'avgprice',avgprice)
    df.insert(6,'market',cur_mv)
    df.insert(7,'sector',cur_sector)
    #for col_label,vol_ind in zip(days_cols_vol,range(cur_vols.shape[1])):
    #    df.insert(len(df.columns),col_label,cur_vols[:,vol_ind])    
    #df.insert(len(df.columns),'pred_day_ret',cur_nextdayrets)
      
    return df

def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=num_workers)(delayed(func)(ind,group) for ind,(name, group) in enumerate(dfGrouped))
    #retLst = extract_data_stock(ind,group) for ind,(name, group) in enumerate(dfGrouped)
    return pd.concat(retLst)

def analyze_date(ind,df):
    print 'ind',ind,'of',len(unq_dates)
    ind = ind + 1
    cur_vals = df[days].values.astype(numpy.float32)

    if len(cur_vals.shape) < 2 or cur_vals.shape[0] == 1:
        return pd.DataFrame()

    means_vec = numpy.mean(cur_vals,axis=0)
    stds_vec = numpy.std(cur_vals,axis=0)
    cur_vals -= means_vec
    cur_vals /= stds_vec   
    df[days] = cur_vals

    cur_rets = df['pred_day_ret'].values
    med_ret = numpy.median(cur_rets)
    classes = (cur_rets > med_ret).astype(numpy.int16) + 1
    avglabel = numpy.mean(classes)
    print '%.2f' % (avglabel)
    df['class_label'] = classes[:,None]
    
    return df
    
start_time = time.time()
home = os.path.expanduser('~')

debug = False
num_days = 250
num_vols = 10
num_workers = 1

if debug: 
    prices_filename = '/media/data2/naamahadad/CSVs/Short.csv'
else:
    prices_filename = '/media/data2/naamahadad/CSVs/1996_2005.csv' if len(sys.argv) < 2 else sys.argv[1]

#link_filename =  home + '/FinData/CSVs/DailyShort.csv'
save_filename = '/media/data2/naamahadad/PyData/1996_2005.hdf' if len(sys.argv) < 3 else sys.argv[2]

print 'prices_filename',prices_filename
print 'save_filename',save_filename

prices = pd.read_csv(prices_filename)#permno,date(yyyymmdd),exchcd,TICKER,VOL,RET
prices = prices.drop(['EXCHCD','PERMNO'],axis=1) #,'VOL'
#prices = prices.drop(prices.columns[[2,3,4],axis=1) 
prices['date'] = pd.to_datetime(prices['date'].values.astype(numpy.str))

prices = prices.sort(columns=['PERMCO','TICKER','date']) # sort by symbol, then by date
#uniqe_stocks = pd.unique(pd.concat((prices['PERMNO'],prices['TICKER']),axis=1))

#if debug: uniqe_stocks = uniqe_stocks[0:50]
final_df = applyParallel(prices.groupby(['PERMCO','TICKER',prices['date'].map(lambda x: x.year)]),extract_data_stock)

##all_dfs = []
##for ind,unq in enumerate(uniqe_stocks):
##    if ind % 100 == 0: print 'cur_unqstock',ind,'of',len(uniqe_stocks) ; sys.stdout.flush()
##    new_df = extract_data_stock(prices[prices.symbol == unq],num_days)
##    all_dfs.append(new_df)
##final_df = pd.concat(all_dfs)
##all_dfs = None
#
##if debug:
##    rand_inds = numpy.random.choice(final_df.shape[0],size=2000,replace=False)
##    mask = numpy.zeros(final_df.shape[0],dtype=numpy.bool)
##    mask[rand_inds] = True
##    final_df = final_df[mask]
#
## add january label
#final_df['pred_day_date'] = pd.to_datetime(final_df['pred_day_date'].values)
#jan_mask = final_df['pred_day_date'].map(lambda x: x.month) == 1
#final_df.insert(len(final_df.columns)-1,'january',jan_mask.astype(numpy.int16))
#
#final_df.insert(len(final_df.columns),'class_label',0)
#days = ['day' + str(i) for i in range(1,num_days)]
#if debug:
#    final_df = final_df[final_df['pred_day_date']>'2015']
#
#unq_dates = final_df['pred_day_date'].unique()
#ind = 0
#if debug:    
#    final_df.groupby('pred_day_date').apply(analyze_date)
##else:
##final_df = applyParallel(final_df.groupby('pred_day_date'),analyze_date)
#
##final_df = dates_df.reset_index()   

print 'done! saving to',save_filename
final_df.to_hdf(save_filename,'table')

tot_time = time.time() - start_time
print 'function took %.2f seconds, %.2f minutes to run' % (tot_time,tot_time/60)
