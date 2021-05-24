# -*- coding: utf-8 -*-
# Questions:
# 1 - when does this data become publicly available to trade on?
# 2- is this data 'as of' or was it later adjusted?
# 3 - why the big gaps in the data, and the big jumps in values?
# 4 - I'm missing the first available data point (2003 mid year, which could flow over into this dataset...)
"""
Created on Thu Oct 20 18:41:25 2016

@author: Richard
"""
############ IMPORTS
from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
#import numpy as np
#import matplotlib.pyplot as plt
import os.path
#from scipy.stats import norm
import backtest_crypto as bt
#from scipy.stats.mstats import gmean
from pycoingecko import CoinGeckoAPI
import datetime
import time


############ INPUTS
#cost = 0.000
frequency = 12.0

cg = CoinGeckoAPI()
datetime1 = datetime.datetime(2021, 1, 1, 10, 20)
datetime2 = datetime.datetime(2021, 5, 15, 10, 20)
from_dateTime = time.mktime(datetime1.timetuple())
to_dateTime = time.mktime(datetime2.timetuple())
#print(from_dateTime)
#print(to_dateTime)
#print(cg.get_price(ids='bitcoin', vs_currencies='usd'))
portfolio = ['bitcoin', 'ethereum', 'solana', 'venus'] #'polkadot', 'cardano', 'chainlink', 'XRP', 'Binance Coin', 'Fantom', 'Matic', 'Aave'
data={}
for name in portfolio:
    try: 
        data[name] = cg.get_coin_market_chart_range_by_id(id=name, vs_currency='usd', from_timestamp=from_dateTime , to_timestamp=to_dateTime )
    except:
        print(name, " not found")

# need to pull the list of tickers from coingecko, and then iterate through them to get all the coins imported and create the database...
#inputs = {}
#inputs.keys = data[name].keys()

for name in portfolio:
    for x in data[name].keys(): # from a dict of lists, create a dict of dataframes
        data[name][x] = pd.DataFrame.from_records(data[name][x], exclude=['Date'], columns=['Date', name], index=['Date'])
        data[name][x] = data[name][x].set_index(pd.to_datetime(data[name][x].index, unit='ms'))
        
        
#inputs = data[name][x].assign(name = data[name][x][name])
inputs = {}
for x in data[name].keys():
    inputs[x] = data[portfolio[0]][x]
    for i in range(len(portfolio)-1):
        inputs[x] = pd.concat([inputs[x], data[portfolio[i+1]][x]], axis=1)

print(inputs)

"""
    df_btc = {}
df_eth = {}



for x in data_eth.keys():
    df_eth[x] = pd.DataFrame.from_records(data_eth[x], exclude=['Date'], columns=['Date', 'ETH'], index=['Date'])
    df_eth[x] = df_eth[x].set_index(pd.to_datetime(df_eth[x].index, unit='ms'))
    merged_df[x] = df_eth[x].assign(BTC = df_btc[x]['BTC'])
"""
#print (df_btc['prices'])
#print (df_eth['prices'])
print(inputs)

"""
#print(cg.get_coin_market_chart_by_id(id='aave', vs_currency='usd', days=14))
for x in cg.get_coins_list():
    try:
        tmp_data = cg.get_coin_market_chart_by_id(id=x['id'], vs_currency='usd', days=14)
        tmp_df = pd.DataFrame(tmp_data['prices'], columns=['Date', x['id']])
        mergedDf = mergedDf.merge(tmp_df)
        print(x['id'], " included in dataframe")
    except:
        print(x['id'], " not found")
"""

inputs_not_to_shift = ['prices','market_caps','total_volumes']
inputs_to_shift=[item for item in inputs if item not in inputs_not_to_shift]

############ RUN CODE
# create a data object 'd' using backtest.py
d = bt.Data(inputs, inputs_to_shift, months_delay_data=3, start=0, delist_value=1)
#print(d.basic_data['prices'])
#data_present = d.accruals()*d.pfd()*d.pman()*d.market_cap()*d.mom()*d.mvi_weight()*d.value()*d.fip()
#data_present = data_present.notnull().astype(int) # would be more reassuring if I had na's here..
#data_present = data_present.replace(0,np.nan)

#print(d.basic_data['prices'].index)
#print(d.basic_data['prices'].columns)

# create a pandas Dataframe of all 1's, with dates for an index and the tickers for column headers
init_pos1 = pd.DataFrame(1, index=d.basic_data['prices'].index, columns=d.basic_data['prices'].columns)
print(init_pos1)
#init_pos1 = init_pos1 * data_present

def run_backtests():
    ret={}  
    mc=bt.Mkt_cap_scr(threshold=1000)
    mw=bt.Mkt_cap_weights()
    
    #print("Creating QM object")
    #QM_object = bt.QM(threshold=1000, scr2_perc = 0.15, scr3_perc = 0.6, upper_limit=0.2)
    #QM_object.backtest(init_pos= init_pos1, data = d) #, rebalance = 'M', #Q-NOV, Q-DEC, Q-OCT                        
    #ret['QM'] = QM_object.calc_ret(price_data= d.basic_data['prices'])
    
    #ret['QM+QV'] = (QM_object.calc_ret(price_data= d.basic_data['prices']) + QV_object.calc_ret(price_data= d.basic_data['prices'])  )/2.0
    
    print("Creating BAH instance")
    BAH_instance = bt.BAH()
    BAH_instance.backtest(mc.run(init_pos1,data=d), data=d)
    print (BAH_instance.backtest(mc.run(init_pos1,data=d), data=d))
    ret['BAH'] = BAH_instance.calc_ret(d.basic_data['prices'])  

    print("Creating mom instance")
    mom_instance = bt.Mom(scr_perc = 0.15)
    mom_instance.run(mc.run(init_pos1,data=d), data=d)
    print (mom_instance.run(mc.run(init_pos1,data=d), data=d))
    ret['mom'] = mom_instance.calc_ret(d.basic_data['prices'])  
    
    #print("Creating fip object") - needs daily price!!
    #fip_object = bt.Fip(scr_perc = 0.6)
    #fip_object.run(mc.run(init_pos1,data=d), data=d)
    #ret['fip'] = fip_object.calc_ret(d.basic_data['prices'])
    
    metrics=(['BAH','mom']) #'QV', 'QV_fin',,'QM','alsi'

    print(" about to plot things")  

    bt.plot_returns(ret, metrics)    
    bt.tabulate_results(ret, metrics, frequency=12.0, risk_free = 0.07) # risk free rate is per annum
    bt.plot_CAGR(ret, metrics, 1)
    bt.plot_CAGR(ret, metrics, 5)

print("About to run_backtests()")
run_backtests()
print("All done")

def QM_mkt_cap(init_pos = init_pos1, threshold = 1000, scr_perc_mom = 0.2, scr_perc_fip = 0.6):  
    mc=bt.Mkt_cap_scr(threshold = threshold)
    mom_object = bt.Mom(scr_perc = scr_perc_mom)
    fip_object = bt.Fip(scr_perc = scr_perc_fip)
    mw = bt.Mkt_cap_weights()
    
    mw.run(fip_object.run(mom_object.run(mc.run(init_pos,data=d), data=d), data=d),data=d)
    return mw

def QM_vmi(init_pos = init_pos1, threshold = 1000, scr_perc_mom = 0.2, scr_perc_fip = 0.6):
    mc=bt.Mkt_cap_scr(threshold = 1000)
    mom_object = bt.Mom(scr_perc = 0.2)
    fip_object = bt.Fip(scr_perc = 0.6)
    mw = bt.Mvi_weights()
    
    mw.run(fip_object.run(mom_object.run(mc.run(init_pos,data=d), data=d), data=d),data=d)
    return mw
    
def QM_parts():
    ret={}  
    #mc=bt.Mkt_cap_scr(threshold=1000)
    
    QM_object = bt.QM(threshold=1000, scr2_perc = 0.2, scr3_perc = 0.6, upper_limit=0.2)
    QM_object.backtest(init_pos= init_pos1, data = d) #, rebalance = 'M', #Q-NOV, Q-DEC, Q-OCT                        
    ret['QM'] = QM_object.calc_ret()
    
#    BAH_object = bt.BAH()
#    BAH_object.backtest(mc.run(init_pos1,data=d), data=d)
#    ret['alsi'] = BAH_object.calc_ret(d.basic_data['prices'])  
#
#    mom_object = bt.Mom(scr_perc = 0.2)
#    mom_object.run(mc.run(init_pos1,data=d), data=d)
#    ret['mom'] = mom_object.calc_ret(d.basic_data['prices'])  
#    
#    fip_object = bt.Fip(scr_perc = 0.6)
#    fip_object.run(mc.run(init_pos1,data=d), data=d)
#    ret['fip'] = fip_object.calc_ret(d.basic_data['prices'])
    
    metrics=(['QM']) #,'fip','QM','alsi'
                                
    bt.plot_returns(ret, metrics)    
    bt.tabulate_results(ret, metrics, frequency=12.0, risk_free = 0.07) # risk free rate is per annum
    bt.plot_CAGR(ret, metrics, 1)
    bt.plot_CAGR(ret, metrics, 5)

# this runs a random screen strategy, as a baseline comparison (as opposed to just comparing to holding the whole universe)
def random_test():
    ret={}  
    mc=bt.Mkt_cap_scr(threshold=0)
    mw = bt.Mkt_cap_weights()
    #mw = bt.Mvi_weights()
    
    #QV_object = bt.QV(threshold=1000, scr2_perc=0.95, scr3_perc = 0.15, scr4_perc = 0.6, upper_limit=0.2)
    #QV_object.backtest(init_pos= init_pos1, data=d) #, rebalance = 'M'                                  
   #ret['QV'] = QV_object.calc_ret(price_data= d.basic_data['prices'])
    
    #QV_fin_object = bt.QV_fin(banks, insurers, threshold=0, scr2_perc = 0.15, scr3_perc = 0.6, upper_limit=0.2)
    #QV_fin_object.backtest(init_pos= init_pos_fin, data=d) #, rebalance = 'M',                                                             
    #ret['QV_fin'] = QV_fin_object.calc_ret(price_data= d.basic_data['prices'])
    
    BAH_object = bt.BAH()
    mw.run(BAH_object.backtest(mc.run(init_pos_non_fins,data=d), data=d), data=d)
    mw.final_positions = bt.limit_pos_size(mw.final_positions)
    ret['alsi'] = mw.calc_ret(d.basic_data['prices'])  
    
    val_object = bt.Value(scr_perc = 0.15)
    mw.run(val_object.run(mc.run(init_pos_non_fins,data=d), data=d), data=d)
    mw.final_positions = bt.limit_pos_size(mw.final_positions)
    ret['value'] = mw.calc_ret(d.basic_data['prices'])
    
    import numpy as np
    rs_obj = bt.Random_screen(scr_perc = 0.15)
    runs=100    
    test=pd.Series(index=range(runs))
    for i in range(runs):
        mw.run(rs_obj.run(mc.run(init_pos_non_fins,data=d)), data=d)
        mw.final_positions = bt.limit_pos_size(mw.final_positions)
        test[i] = ((np.prod(1.+mw.calc_ret(d.basic_data['prices'])))**(frequency/len(mw.calc_ret(d.basic_data['prices']))))-1
    #random_CAGR = test.mean()
    value_CAGR = ((np.prod(1.+ret['value']))**(frequency/len(ret['value'])))-1
    idx = test  < value_CAGR # how many of the random results are less than our value result (CAGR)
    print ("Percentage of random outcomes that are less than our outcome:")
    print ("{:.2%}".format(idx.sum()/runs)) # what percentage of the random results does our value result outperform?
    # need to calculate what % of the random_CAGR's my value_CAGR beats
    metrics=range(10)
    
    #(['value','random','alsi']) #,'QV_fin'
                                
    bt.plot_returns(ret, metrics)    
    bt.tabulate_results(ret, metrics, frequency=12.0, risk_free = 0.07) # risk free rate is per annum
    
    
    bt.plot_CAGR(ret, metrics, 1)
    bt.plot_CAGR(ret, metrics, 5)
    
def testing_QM():
    ret={}  
    
    QM_object={}
    mom_params=[0.05,0.1, 0.15,0.2,0.25]
    fip_params=[0.55,0.6,0.65,0.7,0.75]
    for i in (mom_params):
        for j in (fip_params):
            QM_object[i,j] = bt.QM(threshold=1000, scr2_perc = i, scr3_perc = j, upper_limit=0.2)
            QM_object[i,j].backtest(init_pos= init_pos1, data = d) #, rebalance = 'M', #Q-NOV, Q-DEC, Q-OCT                        
            ret[i,j] = QM_object[i,j].calc_ret(price_data= d.basic_data['prices'])

    metrics=[(x,y) for x in mom_params for y in fip_params]
                        
    #bt.plot_returns(ret, metrics)    
    bt.tabulate_results(ret, metrics, frequency=12.0, risk_free = 0.07) # risk free rate is per annum
    #bt.plot_CAGR(ret, metrics, 1)
    #bt.plot_CAGR(ret, metrics, 5)  
    plotGPR_3d(mom_params, fip_params)
    
def plotGPR_3d(x_params, y_params):
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    # note this: you can skip rows!
    X,Y = np.meshgrid(x_params,y_params)
    X = X.flatten(order='F')
    Y = Y.flatten(order='F') 
    Z = [sum(ret[(x,y)])/abs(sum(ret[(x,y)][ret[(x,y)]<0]))for x in mom_params for y in fip_params] #GPR
    
    xi = np.linspace(min(X),max(X),100)
    yi = np.linspace(min(Y),max(Y),100)
    # VERY IMPORTANT, to tell matplotlib how is your data organized
    zi = griddata((X, Y), Z, (xi[None,:], yi[:,None]), method='cubic')
    
    CS = plt.contour(xi,yi,zi,15,linewidths=0.5,color='k')
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    
    xig, yig = np.meshgrid(xi, yi)
    
    surf = ax.plot_surface(xig, yig, zi,
            linewidth=0)
    
    plt.show()
