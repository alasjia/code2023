'''
该实验采用股票数据data-stock_train.csv
完成任务一
'''
import torch
import torchvision as tv
import numpy as np
import pandas as pd
import math
import matplotlib.pylab as plt
from matplotlib.pylab import style
from scipy.special import logsumexp
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
# from dateutil.parser import parse
from datetime import timedelta

#--------数据准备
stock_data = pd.read_csv("E:/BJTU/其他/时间序列数据分析与挖掘/data-stock_train.csv", index_col=0, parse_dates=[0])
print(stock_data.head())
'''
OUTPUT:
                  High         Low  ...     Volume   Adj Close
Date                                ...                   

2010-01-04  129.226151  127.205109  ...  5896979.0  105.634399
2010-01-05  128.850494  126.972198  ...  4194404.0  105.323608
2010-01-06  127.347855  124.951164  ...  5309229.0  103.657715
2010-01-07  125.612320  121.622841  ...  6498680.0  101.768059
2010-01-08  123.215630  120.285500  ...  4885048.0  100.698906

[5 rows x 6 columns]
'''
#缺失数据Nan填充
data = stock_data.iloc[1:]
data = data.fillna(method = 'ffill')
# print(stock_data.iloc[1:50])
print(data.head())
'''
                  High         Low  ...     Volume   Adj Close
Date                                ...                   

2010-01-05  128.850494  126.972198  ...  4194404.0  105.323608
2010-01-06  127.347855  124.951164  ...  5309229.0  103.657715
2010-01-07  125.612320  121.622841  ...  6498680.0  101.768059
2010-01-08  123.215630  120.285500  ...  4885048.0  100.698906
2010-01-11  123.891808  120.586021  ...  3255763.0  100.257576

[5 rows x 6 columns]
'''
#日期缺失数据填充
# print(stock_data.Close['20100105':'20100120'])
'''
OUTPUT:
Date
2010-01-05    127.302780
2010-01-06    125.289253
2010-01-07    123.005257
2010-01-08    121.712997
2010-01-11    121.179565
2010-01-12    124.455299
2010-01-13    121.292259
2010-01-14    126.626595
2010-01-15    127.625847
2010-01-18    128.459808
2010-01-19    127.550713
2010-01-20    124.658150
Name: Close, dtype: float64
'''
stock_data = stock_data.resample('D').interpolate('linear')
# print(stock_data.Close['20100105':'20100120'])
'''
OUtPUT:
Date
2010-01-05    127.302780
2010-01-06    125.289253
2010-01-07    123.005257
2010-01-08    121.712997
2010-01-09    121.535187
2010-01-10    121.357376
2010-01-11    121.179565
2010-01-12    124.455299
2010-01-13    121.292259
2010-01-14    126.626595
2010-01-15    127.625847
2010-01-16    127.903834
2010-01-17    128.181821
2010-01-18    128.459808
2010-01-19    127.550713
2010-01-20    124.658150
Freq: D, Name: Close, dtype: float64
'''

#--------时序可视化
# plt.plot(stock_data['Close'])
# plt.title('Daily closing price of stock')
# plt.show()


'''
#利用pandas中的DataFrame函数创造一个虚拟的时序数据
data2 = pd.DataFrame(np.random.randn(1000, 1),#随机生成1000行1列的二维数组
                    index= pd.date_range("20230507", periods=1000),#生成时间序列
                    columns=['data2']) #指定列名
print(data2.head())
# data2['data2'] = data2['data2'].cumsum() #逐行相加
# # print(data2.head())
'''

#--------平稳性检验
diff = 0
adf = ADF(stock_data['Close'])
if adf[1] > 0.05:
    print(u'原始序列经检验不平稳，p值为：%s'%(adf[1]))
else:
    print(u'原始序列经检验平稳，p值为：%s'%(adf[1]))
'''
OUTPUT:
原始序列经检验不平稳，p值为：0.9855780932230975
'''

#--------时间序列预处理:纯随机性检验(白噪声检验)
lb = acorr_ljungbox(stock_data['Close'], lags=1)
p = lb.values[0, 1]
if p < 0.05:
	print(u'原始序列为非白噪声序列，p值为:%s' % p)
else:
    print(u'原始序列为白噪声序列，p值为:%s' % p)
'''
OUTPUT:
原始序列为非白噪声序列，p值为:0.0
'''

# #-------确定差分阶数
# #一阶差分处理
# stock_data['diff_1'] = stock_data['Close'].diff(1)
# #二阶差分
# stock_data['diff_2'] = stock_data['diff_1'].diff(1)
# print(stock_data.head())
# # '''
# #创建画板
# fig = plt.figure(1)
# #绘制原始图像
# ax1 = plt.subplot(3,1,1)
# plt.plot(stock_data['Close'])
# plt.title('raw data')
# #绘制一阶差分
# ax1 = plt.subplot(3,1,2)
# plt.plot(stock_data['diff_1'])
# plt.title('diff_1 data')
# #绘制二阶差分
# ax1 = plt.subplot(3,1,3)
# plt.plot(stock_data['diff_2'])
# plt.title('diff_2 data')
# #展示图像
# plt.show()
'''
OUTPUT:
一阶差分与二阶差分效果差距不明显，故选择一阶差分
'''


#-------模型识别
def plotds(xt, nlag=30, fig_size=(12, 10)):
    #如果数据不是pd.Series则进行转换
    if not isinstance(xt, pd.Series):
        xt = pd.Series(xt)
    plt.figure(figsize=fig_size)
    layout = (2, 2)

    #设置画图布局
    ax_xt = plt.subplot2grid(layout, (0, 0), colspan=2)
    ax_acf = plt.subplot2grid(layout, (1, 0))
    ax_pacf = plt.subplot2grid(layout, (1, 1))
    #开始画三类图：原始序列、acf、pacf
    xt.plot(ax=ax_xt)
    ax_xt.set_title('TIME SERIES')
    plot_acf(xt, lags=nlag, ax=ax_acf)  #绘制自相关图
    plot_pacf(xt, lags=nlag, ax=ax_pacf)   #绘制偏自相关图
    plt.tight_layout()
    plt.show()
    return None

# stock_diff = stock_data.diff(1)
# diff = stock_diff.dropna()
# print(diff.head())
# print(diff.dtypes)
# plotds(diff.Close.dropna(), nlag=50)
'''
OUTPUT:
                High       Low  ...     Volume  Adj Close 
Date                            ...
2010-01-05 -0.375656 -0.232910  ... -1702575.0  -0.310791 
2010-01-06 -1.502640 -2.021034  ...  1114825.0  -1.665894 
2010-01-07 -1.735535 -3.328323  ...  1189451.0  -1.889656 
2010-01-08 -2.396690 -1.337341  ... -1613632.0  -1.069153 
2010-01-09  0.225393  0.100174  ...  -543095.0  -0.147110 

[5 rows x 6 columns]
High         float64
Low          float64
Open         float64
Close        float64
Volume       float64
Adj Close    float64
dtype: object

通过查看ACF和PACF图，发现acf与pacf图像均呈现拖尾，
故应采用ARMA模型
'''

# -------利用AIC准则进行模型定阶
# data_df = stock_data.copy()
# aicVal = []
# for ari in range(1, 3):
#     for maj in range(0, 5):
#         try:
#             arima_obj = ARIMA(data_df.Close.tolist(), order=(ari, 0, maj))\
#                 .fit()  #最大似然估计法，可采用method='innovations_mle'
#             aicVal.append([ari, maj, arima_obj.aic])
#         except Exception as e:
#             print(e)
# print(aicVal)
'''
Output:
[[1, 0, 12067.895664296906], 
[1, 1, 12043.504173681984], 
[1, 2, 12045.421828043276], 
[1, 3, 12045.364893455371], 
[1, 4, 12042.20458434857], 
[2, 0, 12043.343868333732], 
[2, 1, 12045.3390943892], 
[2, 2, 12044.295402162967], 
[2, 3, 12043.237697547922], 
[2, 4, 12044.597525482599]]
'''

#-------建立模型
#利用AIC准则，选择使AIC最小的阶数作为ARMA(p,q)的模型阶数，即p=1,q=4
model = ARIMA(stock_data.Close, order=(1, 1, 4))
arima_obj_fin = model.fit()
# print(arima_obj_fin.summary())
'''
注意：
课件中model = ARIMA(stock_data.Close.tolist(), order=(1, 1, 4),)
tolist()会使得model的_index属性为[0,2000]形式，因而无法采用日期进行索引切片
'''
'''
Output:
                               SARIMAX Results            

==============================================================================
Dep. Variable:                      y   No. Observations:                 2553
Model:                 ARIMA(1, 1, 4)   Log Likelihood               -6006.758
Date:                Tue, 19 Sep 2023   AIC                          12025.517
Time:                        23:14:41   BIC                          12060.584
Sample:                             0   HQIC                         12038.235
                               - 2553                     

Covariance Type:                  opg                     

==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1          0.7245      0.144      5.019      0.000       0.442       1.007
ma.L1         -0.6233      0.145     -4.294      0.000      -0.908      -0.339
ma.L2         -0.0707      0.023     -3.125      0.002      -0.115      -0.026
ma.L3          0.0181      0.019      0.931      0.352      -0.020       0.056
ma.L4         -0.0563      0.018     -3.187      0.001      -0.091      -0.022
sigma2         6.4857      0.080     81.436      0.000       6.330       6.642
===================================================================================
Ljung-Box (L1) (Q):                   0.01   Jarque-Bera (JB):              8833.27
Prob(Q):                              0.94   Prob(JB):                         0.00
Heteroskedasticity (H):               3.41   Skew:                             0.86
Prob(H) (two-sided):                  0.00   Kurtosis:                        11.95
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
'''

# #-------参数估计
# #AR(p)、MA(q)、ARMA(p,q)模型的估计方法较多，大体上分为3类： 
# # 最小二乘估计、矩估计和利用自相关函数的直接估计。

# #-------模型拟合与预测
# #模型拟合:真实数据图像与拟合数据图像对比
# stock_data['ARIMA'] = arima_obj_fin.predict()
# f, axarr = plt.subplots(1, sharex = True)
# f.set_size_inches(12, 8)
# stock_data['Close'].iloc[len(stock_data) - 100:]\
# .plot(color = 'b', linestyle = '-', ax = axarr)   #真实数据图像
# stock_data['ARIMA'].iloc[len(stock_data) - 100:]\
# .plot(color = 'r', linestyle = '--', ax = axarr)  #拟合数据图像
# axarr.set_title('ARIMA(1,1,4)')
# plt.xlabel('index')
# plt.ylabel('close price')
# plt.show()
# # 模型预测
# plot_predict(arima_obj_fin, len(stock_data)-50, len(stock_data)+10)
# plt.show()
# # 数值结果展示
# predict = arima_obj_fin.predict(start=1, end=len(stock_data)+10)
# print(predict[len(predict)-10:]) 
# '''
# [318.88247575 318.88247573 318.88247572 318.88247571 318.88247571
#  318.8824757  318.8824757  318.8824757  318.8824757  318.8824757 ]
#  '''

# 对部分预测结果进行可视化展示
# output1 = arima_obj_fin.predict('20120201', '20120207', dynamic=True, type='levels')  #numpy.array
# stock_forcast = pd.concat([stock_data.Close['20120115':'20120207'], output1], axis=1, keys=['original', 'predicted'])
# plt.figure()
# plt.plot(stock_forcast)
# plt.title('Original vs predicted')
# plt.show()


# #预测后续数据
# # 选择可视化的原始数据区间
# true_data2 = stock_data.Close['2016/12/15': '2016/12/30'] 
# fig, ax = plt.subplots()
# ax = true_data2.loc['2016/12/15': ].plot(ax=ax) # 设置坐标轴起点
# plot_predict(arima_obj_fin, '2016/12/25', '2017/01/30', ax=ax) # 可视化预测时间起点与终点
# plt.show()

# #-------模型评估：平均绝对误差MAE，均方误差MSE，均方根误差RMSE
# #对短期预测结果进行评估
# short_label = stock_data.Close['20120201': '20120202']
# short_prediction = output1[:2]
# short_mse_score = mse(short_label, short_prediction)
# short_rmse_score = math.sqrt(short_mse_score)
# short_mae_score = mae(short_label, short_prediction)
# print('short_MSE: %.4f, short_RMSE: %.4f, short_MAE: %.4f' % (short_mse_score, short_rmse_score, short_mae_score))
# #对长期预测结果进行评估
# long_label = stock_data.Close['20120201': '20120207']
# long_prediction = output1
# long_mse_score = mse(long_label, long_prediction)
# long_rmse_score = math.sqrt(long_mse_score)
# long_mae_score = mae(long_label, long_prediction)
# print('long_MSE: %.4f, long_RMSE: %.4f, long_MAE: %.4f' % (long_mse_score, long_rmse_score, long_mae_score))
# '''
# OUTPUT:
# short_MSE: 0.1228, short_RMSE: 0.3505, short_MAE: 0.3335
# long_MSE: 0.6420, long_RMSE: 0.8013, long_MAE: 0.6463
# '''