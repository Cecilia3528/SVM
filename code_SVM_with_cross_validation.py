#SVM & Cross-Validation
import tushare as ts
import numpy as np
import pandas as pd
import talib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def train_data(data_train,data_test,C,kernel): 
    model_C = C
    model_kernel = kernel
    X_train = data_train[['ema','stddev','slope','rsi','wr']].values
    y_train = data_train['rise'].values
    X_test = data_test[['ema','stddev','slope','rsi','wr']].values
    y_test = data_test['rise'].values
    #standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    classifier = SVC(C=model_C,kernel = model_kernel) 
    classifier.fit(X_train, y_train)
    return classifier.predict(X_train), classifier.predict(X_test),y_test
    
def valuation(list_of_y_predicts):
    y_pred = list_of_y_predicts[1]
    y = list_of_y_predicts[2]
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    df_valuation = pd.DataFrame(y_pred,y) # y is index
    accuracy = df_valuation[df_valuation.index==df_valuation[0]].shape[0]/df_valuation[0].shape[0]
    precision = df_valuation[(df_valuation[0] == 1) & (df_valuation.index== 1)].shape[0]\
                /df_valuation[df_valuation[0] ==1].shape[0]
    return rmse, accuracy, precision


#get data
data = ts.get_k_data(code='hs300', start = '2015-04-08', end = '2023-11-05', ktype = 'D')
data = data.set_index('date')
data.index = pd.to_datetime(data.index)
data = data[['open','close','high','low']]

#process data
data['ema'] = talib.EMA(data['close'].values, timeperiod = 20) #exponential moving average
data['stddev'] = talib.STDDEV(data['close'].values, timeperiod = 20, nbdev = 1) 
data['slope'] = talib.LINEARREG_SLOPE(data['close'].values, timeperiod = 5)
data['rsi'] = talib.RSI(data['close'].values, timeperiod = 14)
data['wr'] = talib.WILLR(data['high'].values, data['low'].values, data['close'].values, timeperiod = 7)
data['pct'] = data['close'].shift(-1)/data['close']-1
data['rise'] = data['pct'].apply(lambda x : 1 if x>0 else 0)
data = data.dropna()

#Build model
#evaluate kernels

num_train = round(len(data)*0.8)
data_train = data.iloc[:num_train, :]
data_test = data.iloc[num_train:,:]

kf = KFold(n_splits=4,shuffle=True)
i=0
evaluation_kernel_dic = {'kernel':[],'rmse':[],'accuracy':[],'precision':[]}
kernel_list=['linear','poly','rbf','sigmoid']
for train_index , test_index in kf.split(data_train):  
    data_sub_train = data_train.iloc[train_index]
    data_sub_test = data_train.iloc[test_index]
    C = 2
    kernel = kernel_list[i]
    rmse, accuracy, precision = valuation(train_data(data_sub_train,data_sub_test,C,kernel))
    evaluation_kernel_dic['kernel']+=[kernel]
    evaluation_kernel_dic['rmse']+=[rmse]
    evaluation_kernel_dic['accuracy']+=[accuracy]
    evaluation_kernel_dic['precision']+=[precision]
    i=i+1
    
eva_k_df = pd.DataFrame(evaluation_kernel_dic)
eva_k_df['eva']=eva_k_df[['rmse','precision']].mean(axis=1)
best_kernel=eva_k_df.sort_values(by='eva').iloc[-1,0]

#evaluate penaly coefficient
kf_C = KFold(n_splits=10,shuffle=True)
i=0
evaluation_C_dic = {'C':[],'rmse':[],'accuracy':[],'precision':[]}
C_list = np.arange(1,6,0.5)
for train_index , test_index in kf_C.split(data_train):  
    data_sub_train = data_train.iloc[train_index]
    data_sub_test = data_train.iloc[test_index]
    kernel = best_kernel
    C = C_list[i]
    rmse, accuracy, precision = valuation(train_data(data_sub_train,data_sub_test,C,kernel))
    evaluation_C_dic['C']+=[C]
    evaluation_C_dic['rmse']+=[rmse]
    evaluation_C_dic['accuracy']+=[accuracy]
    evaluation_C_dic['precision']+=[precision]
    i=i+1
    
eva_C_df = pd.DataFrame(evaluation_C_dic)
eva_C_df['eva']=eva_C_df[['rmse','precision']].mean(axis=1)
best_C=eva_C_df.sort_values(by='eva').iloc[-1,0]
print('The best kernel is {} and the best penaly coefficient is {}'.format(best_kernel, best_C))
print('*'*40)

#stimulation (without transaction costs)
data_test['pred'] = train_data(data_train, data_test, best_C, best_kernel)[1]
data_test['previous']=train_data(data_train, data_test, 2, 'rbf')[1]

data_test['strategy_pct'] = data_test.apply(lambda x: x.pct if x.pred>0 else -x.pct, axis=1)
data_test['strategy'] = (1.0 + data_test['strategy_pct']).cumprod()

data_test['previous_strategy_pct'] = data_test.apply(lambda x: x.pct if x.previous>0 else -x.pct, axis=1)
data_test['previous_strategy'] = (1.0 + data_test['previous_strategy_pct']).cumprod()

data_test['hs300'] = (1.0 + data_test['pct']).cumprod()
annual_return = 100 * (pow(data_test['strategy'].iloc[-1], 250/data_test.shape[0]) - 1.0)
annual_return_p = 100 * (pow(data_test['previous_strategy'].iloc[-1], 250/data_test.shape[0]) - 1.0)
print('Annual return for 沪深300 with SVM selection is：%.2f%%' %annual_return)
print('Annual return for 沪深300 with SVM selection is（random strategy）：%.2f%%' %annual_return_p)
ax = data_test[['strategy','hs300','previous_strategy']].plot(title='Selection of 沪深300 using SVM')
plt.show()
