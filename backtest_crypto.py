# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 10:27:01 2016

@author: Richard
"""
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import os.path
from scipy.stats import norm

############ FUNCTIONS
# calculates maximum drawdown of a backtest
def max_dd(ser):
    max2here = ser.cummax()
    dd2here = (ser - max2here)/max2here
    dd2here[dd2here == -np.inf] = 0
    return dd2here.min()

# cleans data by removing a nan in a row with the mean of that row
def data_clean(df): 
    df = df.T.fillna(df.mean(axis=1)).T # replace nans in rows with the mean of that row - only if it isn't a row of nan's
    df = df.interpolate() # interpolate where there are values before and after
    #df = df.fillna(method='bfill') # OPTIONAL: fill backwards where there are no values going back...
    return df
        
# reads in data from a .csv file, with some quite specific changes to fix the idiosyncracies of the dataset I was using
def read_in_data(filename):
    df = pd.read_csv(filename)
    try:
        df = df.drop_duplicates()
    except:
        pass
    df = df.dropna(axis=1,how='all') # get rid of columns with all nans
    df = df.dropna(axis=0,how='all') # get rid of rows with all nans
    df = df.set_index('Company')
    df = df.transpose()
    #df = df.fillna(0)
    try:
        df=df.drop(['42671']) # this is for the daily data - weirdly adds in this extra row / column
    except:
        pass
    try:
        df=df.drop(['42675'])
    except:
        pass
    df.index=pd.to_datetime(df.index.astype(str), format='%d/%m/%Y')
    try:
        del df['STP-JSE'] # there's a woopsie in the price data for this stock, so I've just excluded it for now...
    except:
        pass
    try:
        del df['WES-JSE'] # there's a woopsie in the price data for this stock, so I've just excluded it for now...
    except:
        pass
    return df

# screens out the bottom 10% of the dataset - on what basis?
def screen(init_pos, data_in, scr_perc=0.10, ascending=True):
    tmp_data=data_in*init_pos
    tmp_data=tmp_data.replace(0,np.nan)
    tmp_data=tmp_data.values  
    if ascending == True:   
        perc=np.nanpercentile(tmp_data,100*scr_perc,axis=1, keepdims=True)    
        screen=np.where(tmp_data<perc,1,0)
    elif ascending == False:
        perc=np.nanpercentile(tmp_data,100-100*scr_perc,axis=1, keepdims=True)    
        screen=np.where(tmp_data>perc,1,0)
    screen = pd.DataFrame(screen, index=init_pos.index, columns = init_pos.columns)
    return screen


#def random_screen(init_pos, scr_perc=0.10):
#    random_df = pd.DataFrame(np.random.rand(init_pos.shape[0],init_pos.shape[1]), index=init_pos.index, columns = init_pos.columns) #np.random.randint(0,init_pos.shape[1],size=init_pos.shape), index=init_pos.index, columns = init_pos.columns)    
#    tmp_data=random_df*init_pos
#    tmp_data=tmp_data.replace(0,np.nan)
#    data_count = tmp_data.count(axis=1)
#    tmp_screen = tmp_data.rank(axis=1, ascending=True) # change this to generate random numbers
#    data_list = tmp_screen.columns.tolist()
#    for i in data_list:
#        tmp_screen[i][tmp_screen[i]>data_count*scr_perc]=0
#    tmp_screen[tmp_screen>0]=1
#    tmp_screen = tmp_screen.fillna(0)
#    return tmp_screen

# this boolean screen just returns a 1 or 0 in each position indicating whether that ticker is in/out on that day
def bool_screen(init_pos, data_in, threshold): 
    tmp_data=data_in*1    
    tmp_data[tmp_data<threshold]=0
    tmp_data[tmp_data>=threshold]=1
    tmp_screen = tmp_data.fillna(0)
    positions = tmp_screen*init_pos
    return positions

# this screen attributes a weight to each position, according to some kind of weighting data (market cap or inverse volatility)
def weight(positions, weighting_data):
    positions = ((positions*weighting_data).T/(positions*weighting_data).sum(axis=1)).T
    return positions

# limits the size of any single position in the portfolio from going above an upper limit or below an lower limit
def limit_pos_size(weights, lower_limit=0.02, upper_limit = 0.2):     
    weights = weights.clip(upper=upper_limit)
    row_sum = weights.sum(axis=1)
    row_sum_2 = weights[weights!=upper_limit].sum(axis=1)
    new_weights = (weights.T*(1+(1-row_sum)/(row_sum_2))).T
    new_weights = new_weights.clip(upper=upper_limit)
    for j in range(5):
        for i in range(5): #while new_weights.sum(axis=1)<1:
            row_sum = new_weights.sum(axis=1)
            row_sum_2 = new_weights[new_weights!=upper_limit].sum(axis=1)
            new_weights = (new_weights.T*(1+(1-row_sum)/(row_sum_2))).T
            new_weights = new_weights.clip(upper=upper_limit)
        new_weights = new_weights.where(new_weights>lower_limit,0)
    
    return new_weights   

# calculates the return of the portfolio
def calc_ret(positions, prices, cost=0): #reinsert cost here!!
    pnl = positions.shift(1)*(prices-prices.shift(1))/prices.shift(1) # calculate pnl as position yesterday x price change since yesterday
    pnl=pnl.fillna(0)
    #pnl[positions.shift(1).fillna(0)!=positions.shift(2).fillna(0)]-= cost  #subtract transaction costs:
    total_pnl=pnl.sum(axis=1) # sum across all tickers to get total pnl per day
    total_positions=positions.sum(axis=1) # sum across tickers to get total number of positions
    ret=total_pnl / total_positions.shift(1) # divide pnl by total weight of position in market to get return
    ret[ret==-np.inf]=0 # zero out the infs - a problem creeps in because of 27/4/05??
    ret=ret.fillna(0)
    return ret

# not sure what this is?
def trend_filter(j203_price, ret, risk_free): # all three of these are series
    tmp=pd.concat([j203_price,pd.rolling_mean(j203_price, 12)],axis=1) # create a temp dataframe with alsi price & 12 mth MA
    tmp=tmp.reindex(index=ret.index,method='nearest')
    tmp['diff']=tmp.iloc[:,0]-tmp.iloc[:,1] # diff btw price & MA
    tmp=tmp.shift(1)
    tmp[tmp>0]=0
    tmp[tmp<0]=1 # we generate a signal when last months price below MA
    risk_free=risk_free.reindex(index=tmp.index, method='nearest')
    tmp['returns']=tmp['diff']*risk_free # in months where there's a signal, we take the risk-free rate
    ret[tmp['diff']==1]=tmp['returns'] # we insert the risk-free return into the ret dataframe

# plot the returns out output the plot
def plot_returns(ret, metrics):
    fig, ax1 = plt.subplots()
    colors = ['red','blue','green','magenta','pink','orange', 'purple','yellow','black','cyan','turquoise','white']
    for i in range(len(metrics)):
        plt.plot(100*np.cumprod(1.+ret[metrics[i]]), color=colors[i], lw=1, label = metrics[i])        
    #plt.yscale('log')
    plt.grid(True)
    plt.legend(loc=0)
    ax1.ylabel = ('return')
    plt.xlabel('date')
    ax2 = ax1.twinx()
    #plt.plot(QV_object.final_positions[QV_object.final_positions>0].count(axis=1),'ro') # this is the number of stocks
    #plt.ylabel('no of stocks')
    plt.title('Value of $100 invested (log scale)')   
    plt.show()

# plot the compound annual growth rate of the portfolio    
def plot_CAGR(ret, metrics, num_years):
    rolling_CAGR={}
    colors = ['red','blue','green','magenta','pink','orange', 'purple','yellow','black','cyan','turquoise','white']
    for i in range(len(metrics)):
        rolling_CAGR[metrics[i]] = ret[metrics[i]].rolling(num_years*12).apply(lambda x: (np.prod(1+x)**(12.0/len(x))-1))
        plt.plot(rolling_CAGR[metrics[i]], color=colors[i], lw=1, label = metrics[i])
    plt.grid(True)
    plt.legend(loc=2)
    plt.title('{} yr rolling CAGR'.format(num_years)) 
    plt.show()

# tabulate the results and output the table
def tabulate_results(ret, metrics, frequency = 365.0, risk_free = 0.07):  
    MAR=(1+risk_free)**(1.0/frequency)-1 # an annual rate of 7% converted to a monthly rate
    table_list = [[metrics[i], 
                    "{:.2%}".format(((np.prod(1.+ret[metrics[i]]))**(frequency/len(ret[metrics[i]])))-1), #APR
                    "{:.2%}".format(np.sqrt(frequency)*np.std(ret[metrics[i]])),
                    "{:.3}".format(np.sqrt(12)*(ret[metrics[i]].mean()-MAR)/np.std(ret[metrics[i]])), #Sharpe - need to deduce risk-free rate
                    "{:.3}".format(np.sqrt(frequency)*(np.mean(ret[metrics[i]])-MAR)/np.sqrt(sum(((ret[metrics[i]]-MAR)[(ret[metrics[i]]-MAR)<0])**2/len((ret[metrics[i]]-MAR)[(ret[metrics[i]]-MAR)<0])))),
                    "{:.3}".format(sum(ret[metrics[i]])/abs(sum(ret[metrics[i]][ret[metrics[i]]<0]))), #
                    "{:.2%}".format(max_dd(np.cumprod(1.+ret[metrics[i]]))),
                    "{:.2%}".format(max(ret[metrics[i]])),
                    "{:.2%}".format(min(ret[metrics[i]])),
                    "{:.2%}".format(float(ret[metrics[i]][ret[metrics[i]]>0].count())/ret[metrics[i]].count())] # max drawdown - need to make it a percentage
                    for i in range(len(metrics))]
    print (tabulate(table_list, headers=['CAGR','Std Dev', 'Sharpe','Sortino','GPR','Max Drawdown','Best mth', 'Worst mth','Win mths'])) #,floatfmt=".2%"

# NB - need to assume a price at which we would have exited delisted stocks
def delisting(prices, delist_value): # delist value here can be set as a percentage of final price
    prices = prices.replace(0,np.nan)
    columns= prices.columns
    for i in range(len(columns)):
        try:
            row_number = prices.index.get_loc(prices[columns[i]].last_valid_index()) # get ordinal location of the index of the last valid data point in the column
            prices[columns[i]][row_number+1] = prices[columns[i]][row_number]*delist_value
        except:
            pass
    return prices
############## Classes ####################

# a general data object
class Data(object):
    
    def __init__(self, inputs, inputs_to_shift, months_delay_data=3, start=0, delist_value=0.5):        
        self.months_delay_data = months_delay_data # should this be here, couldn't they be inputs??
        self.start = start # should this be here, couldn't they be inputs??
        self.delist_value = delist_value
        self.inputs = inputs
        self.inputs_to_shift = inputs_to_shift
        #self.path = path
        self.basic_data = {}
    
        self.basic_data['prices'] = inputs['prices']
        self.basic_data['market_caps'] = inputs['market_caps']
        self.basic_data['total_volumes'] = inputs['total_volumes']
        """
        for i in range(len(self.inputs)):
            self.basic_data[self.inputs[i]] = read_in_data(os.path.join(self.path, self.inputs[i]+'.csv'))
        
        data_to_shift = self.inputs_to_shift
        for i in range(len(data_to_shift)):
            self.basic_data[data_to_shift[i]] = self.basic_data[data_to_shift[i]].shift(months_delay_data)
        
        data_to_clip = ['bps','ebit_oper','entrpr_val', 'ebit_oper']
        for i in range(len(data_to_clip)):
            self.basic_data[data_to_clip[i]] = np.clip(self.basic_data[data_to_clip[i]],1,100000000) # this clipping takes care of negative values
    
        self.index = self.basic_data['prices'].index
        self.index = self.index[start:] # we just go from the date when the valuation data starts...
        for i in range(len(self.inputs)):
            self.basic_data[self.inputs[i]]= self.basic_data[self.inputs[i]].reindex(index=self.index, method='nearest')
            
        ############## HANDLING DELISTINGS
        self.basic_data['prices']= delisting(self.basic_data['prices'],delist_value)
        
        self.daily_price = read_in_data(os.path.join(self.path,'other data','price_daily.csv')).replace(0,np.nan)
        
        self.j203_price = read_in_data(os.path.join(self.path,'other data','j203_price.csv')) # this is monthly
        self.j203_price = self.j203_price.reindex(index=self.index, method='nearest') 
        
        self.risk_free =  pd.read_csv(os.path.join(self.path,'other data','risk_free_rate.csv'))
        self.risk_free = self.risk_free.dropna(axis=1,how='all') # get rid of columns with all nans
        self.risk_free = self.risk_free.dropna(axis=0,how='all') # get rid of rows with all nans
        self.risk_free = self.risk_free.set_index('Unnamed: 0')
        self.risk_free = self.risk_free.transpose()
        self.risk_free.index=pd.to_datetime(self.risk_free.index.astype(str), format='%d/%m/%Y')
        self.risk_free = self.risk_free['TRYZA10Y-FDS']
        self.risk_free = self.risk_free.reindex(index=self.index, method='nearest')
        self.risk_free = self.risk_free.fillna(method='pad')
        self.risk_free=self.risk_free.astype(float)
        self.risk_free = self.risk_free-2
        self.risk_free=self.risk_free/100
        self.risk_free = self.risk_free.apply(lambda x: (x+1)**(1.0/12)-1)
        """

# a general strategy object
class Strategy(object):
    
    def liquidity(self, data):
        liquidity = data.basic_data['volume_monthly']/data.basic_data['free_float']
        return liquidity
    
    def market_cap(self, data):
        return data.basic_data['market_caps']
    
    def equal_weight(self, data):
        equal_weight = pd.DataFrame(1, index=data.basic_data['market_caps'].index, columns=data.basic_data['market_caps'].columns)
        return equal_weight
         
    def mkt_weight(self, data):
        mkt_weight = data.basic_data['market_caps']
        return mkt_weight
        
    def mvi_weight(self, data, mvi_window_len=220):
        stdev = pd.rolling_std(data.daily_price, window=mvi_window_len)
        mvi_weight = 1/stdev #daily_price/stdev - this is the volatility indicator
        #test=daily_price.diff()
        #downside_dev = pd.rolling_apply(test, 110, lambda x: np.sqrt((x[x<0]-x.mean())**2).sum()/len(x[x<0]) )
        mvi_weight= mvi_weight.reindex(index=data.index, method='nearest')    
        return mvi_weight
        
    def set_weights(self,data, mvi_window_len=220):
        weighting = {}
        weighting['equal'] = pd.DataFrame(1, index=data.basic_data['market_caps'].index, columns=data.basic_data['market_caps'].columns)
        weighting['mkt_cap'] = data.basic_data['market_caps']
        stdev = pd.rolling_std(data.daily_price, window=mvi_window_len)
        mvi_weight = 1/stdev #daily_price/stdev - this is the volatility indicator
        #test=daily_price.diff()
        #downside_dev = pd.rolling_apply(test, 110, lambda x: np.sqrt((x[x<0]-x.mean())**2).sum()/len(x[x<0]) )
        weighting['mvi'] = mvi_weight.reindex(index=data.index, method='nearest')
        return weighting
        
    def calc_ret(self, price_data, start_date='2000-01-30'):
        #ret = calc_ret(self.final_positions.ix[start_date:], price_data.ix[start_date:]) - .ix has been deprecated
        ret = calc_ret(self.final_positions.loc[start_date:], price_data.loc[start_date:])
        return ret
        
    def latest(self, date='2016-11-01'):
        #latest = self.final_positions.ix[date][self.final_positions.ix[date]>0] - .ix has been deprecated
        latest = self.final_positions.loc[date][self.final_positions.loc[date]>0]
        return latest  

# Quantitative value screen - which has many components, many of which are below
class QV(Strategy):
    
    def __init__(self, threshold=1000, scr2_perc=0.95, scr3_perc = 0.15, scr4_perc = 0.6, weighting='mkt_cap', upper_limit=0.2): #, trend_filter=False, threshold = 0
        #self.rebalance = rebalance
        self.threshold = threshold
        self.scr2_perc = scr2_perc# forensics -> positions 2
        self.scr3_perc = scr3_perc # value -> positions 3
        self.scr4_perc = scr4_perc # quality -> positions 4
        self.upper_limit = upper_limit
        self.weighting = weighting
        
    def backtest(self, init_pos, data):
        
        self.positions1 = bool_screen(init_pos, self.market_cap(data), threshold=self.threshold) # mkt cap of R2bn or more
        
        accruals_object = Accruals(scr_perc = self.scr2_perc)
        self.positions2a = accruals_object.run(self.positions1,data)
        
        pman_object = Pman(scr_perc = self.scr2_perc)
        self.positions2b = pman_object.run(self.positions1,data)
        
        pfd_object = Pfd(scr_perc = self.scr2_perc)
        self.positions2c = pfd_object.run(self.positions1,data)
        
        self.positions2 = self.positions2a*self.positions2b*self.positions2c # combine the results of pfd & accrualspositions2 = screen(positions1, data_for_scr2, scr_perc = scr2_perc, ascending=scr2_asc)
        
        value_object = Value(scr_perc = self.scr3_perc)
        self.positions3 = value_object.run(self.positions2, data)
        
        quality_object = Quality(scr_perc=self.scr4_perc)
        self.positions4 = quality_object.run(self.positions3, data)
        
        #mw_object=mkt_cap_weights()
        mw_object=Mvi_weights()
        self.positions5 = mw_object.run(self.positions4,data) #weight(self.positions4, self.set_weights(data)[self.weighting])

        self.final_positions = limit_pos_size(self.positions5, self.upper_limit) 
        
# Quantitative momentum strategy        
class QM(Strategy):
    
    def __init__(self, threshold=1000, scr2_perc = 0.15, scr3_perc = 0.6, upper_limit=0.2):
        self.threshold = threshold
        self.scr2_perc = scr2_perc # momentum
        self.scr3_perc = scr3_perc # fip - quality of momentum  
        self.upper_limit = upper_limit
        
    def backtest(self, init_pos, data):     
        mc= Mkt_cap_scr(threshold=self.threshold)
        self.positions1 = mc.run(init_pos, data) #bool_screen(init_pos, self.market_cap(data), self.threshold) # mkt cap of R2bn or more threshold = 2000
        # momentum
        m=Mom(scr_perc = self.scr2_perc)
        self.positions2 = m.run(self.positions1, data)
        
        f=Fip(scr_perc = self.scr3_perc)
        self.positions3 = f.run(self.positions2, data)

        self.positions4 = weight(self.positions3, self.mvi_weight(data))

        self.final_positions = limit_pos_size(self.positions4)
        
        return self.final_positions

# Momentum strategy
class Mom(Strategy):
    
    def __init__(self, scr_perc = 0.15):
        self.scr_perc=scr_perc 
        
    def run(self, init_pos, data):     
        mom = data.basic_data['prices'].pct_change(11).shift(1) #pd.rolling_apply(basic_data['prices'].pct_change(), 12, lambda x: np.prod(1 + x) - 1)
        #mom = data_clean(mom)      
        self.final_positions = screen(init_pos, mom, scr_perc = self.scr_perc, ascending=False)
        return self.final_positions

# FIP
class Fip(Strategy):
    
    def __init__(self, scr_perc = 0.6):
        self.scr_perc = scr_perc # fip - quality of momentum  
        
    def run(self, init_pos, data):     
        #fip = np.sign(data.basic_data['prices'].pct_change(11).shift(1))*pd.rolling_apply( data.daily_price.pct_change(), 252, lambda x: (len(np.where(x<0)[0])-len(np.where(x>0)[0]))/252.0)
        fip = np.sign(data.basic_data['prices'].pct_change(11).shift(1))*data.daily_price.pct_change().rolling(252).apply(lambda x: (len(np.where(x<0)[0])-len(np.where(x>0)[0]))/252.0)
        fip = fip.reindex(index=data.index, method='ffill')   
        self.final_positions = screen(init_pos, fip, scr_perc = self.scr_perc, ascending=True)      
        return self.final_positions 
 
# Ken Long strategy
class Kl_str(Strategy):
    
    def __init__(self, scr_perc = 0.15):
        self.scr_perc=scr_perc
        
    def run(self, init_pos, data):     
        kl_str = 0.7*data.daily_price.pct_change(13*5)+0.3*data.daily_price.pct_change(26*5)
        kl_str = (kl_str.rank(axis=1, ascending = True).T/kl_str.count(axis=1)).T # take percentile
        kl_str= kl_str.reindex(index=data.index, method='nearest')      
        self.final_positions = screen(init_pos, kl_str, scr_perc = self.scr_perc, ascending=False)
        return self.final_positions

# Ken Long conservative strategy
class Kl_con(Strategy):
  
    def __init__(self, scr_perc = 0.6):
        self.scr_perc=scr_perc
        
    def run(self, init_pos, data):     
        kl_str_obj = Kl_str()        
        kl_con = 2*kl_str_obj.run(init_pos, data) + 2*kl_str_obj.run(init_pos, data).shift(1*5)+1.75*kl_str_obj.run(init_pos, data).shift(2*5)+1.75*kl_str_obj.run(init_pos, data).shift(3*5)+1.5*kl_str_obj.run(init_pos, data).shift(4*5)+1.5*kl_str_obj.run(init_pos, data).shift(5*5)+1.25*kl_str_obj.run(init_pos, data).shift(6*5)+1.25*kl_str_obj.run(init_pos, data).shift(7*5)+1*kl_str_obj.run(init_pos, data).shift(8*5)+1*kl_str_obj.run(init_pos, data).shift(9*5)
        # = pd.rolling_mean(data.kl_str(), 10) 
        kl_con = (kl_con.rank(axis=1, ascending = True).T/kl_con.count(axis=1)).T # take percentile
        kl_con= kl_con.reindex(index=data.index, method='nearest')     
        self.final_positions = screen(init_pos, kl_con, scr_perc = self.scr_perc, ascending=False)
        return self.final_positions

# Ken Long quality strategy
class Kl_qual(Strategy):
        
    def __init__(self, scr_perc = 0.6):
        self.scr_perc=scr_perc
        
    def run(self, init_pos, data):     
        weekly_price = data.daily_price.asfreq('W','nearest')        
        kl_qual = pd.rolling_mean(weekly_price.pct_change(1), 40)/pd.rolling_std(weekly_price.pct_change(1), 40) 
        kl_qual = (kl_qual.rank(axis=1, ascending = True).T/kl_qual.count(axis=1)).T# take percentile
        kl_qual= kl_qual.reindex(index=data.index, method='nearest')    
        self.final_positions = screen(init_pos, kl_qual, scr_perc = self.scr_perc, ascending=False)
        return self.final_positions

# screen out companies with a market cap lower than the threshold (in millions)   
class Mkt_cap_scr(Strategy):
    
    def __init__(self, threshold=1000):
        self.threshold=threshold 
        
    def run(self, init_pos, data):     
        self.final_positions = bool_screen(init_pos, data.basic_data['market_caps'], self.threshold) # mkt cap of R2bn or more threshold = 2000
        return self.final_positions       

# weight the stocks by market cap within the strategy
class Mkt_cap_weights(Strategy):
        
    def run(self, init_pos, data):     
        self.final_positions = weight(init_pos, self.mkt_weight(data))
        return self.final_positions 

# weight stocks by inverse volatility
class Mvi_weights(Strategy):
        
    def run(self, init_pos, data, mvi_window_len=220):     
        #stdev = pd.rolling_std(data.daily_price, window=mvi_window_len) - rolling_std has been deprecated in pandas
        stdev = data.daily_price.rolling(mvi_window_len).std()
        mvi_weight = 1/stdev #daily_price/stdev - this is the volatility indicator
        #test=daily_price.diff()
        #downside_dev = pd.rolling_apply(test, 110, lambda x: np.sqrt((x[x<0]-x.mean())**2).sum()/len(x[x<0]) )
        mvi_weight= mvi_weight.reindex(index=data.index, method='nearest')        
        self.final_positions = weight(init_pos, mvi_weight)
        return self.final_positions

# BAH
class BAH(Strategy):
    def __init__(self):
        pass
        
    def backtest(self, init_pos, data):     
        
        self.final_positions = weight(init_pos, self.mkt_weight(data))
        
        return self.final_positions

# a random screen / strategy, for comparative purposes
class Random_screen(Strategy):
        
    def __init__(self, scr_perc = 0.15):
        self.scr_perc=scr_perc
        
    def run(self, init_pos):     
        self.final_positions = random_screen(init_pos, scr_perc = self.scr_perc)
        return self.final_positions
        
def plot_print(ret, label):
    cum_ret = np.cumsum(ret)
    plt.plot(cum_ret, label=label)

# channeling strategy
class Channeling(Strategy):
        
    def __init__(self, scr_perc = 0.15):
        self.scr_perc=scr_perc
        
    def run(self, init_pos):     
        will_r = talib.WILLR(high, low, close)
        self.final_positions = np.where(will_r>close.shift(1),1,0) # something like this - this is very basic
        return self.final_positions
