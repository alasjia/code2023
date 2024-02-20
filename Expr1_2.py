'''
该实验采用Google数据集
完成任务二、三
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
'''
数据字段说明：
lastSyncTime：上次同步时间；时间变量；
steps：步数；连续变量；
steps：距离；连续变量；
runDistance：跑步距离；连续变量；
calories：卡路里；连续变量；
'''
google_data = pd.read_csv("E:/BJTU/其他/时间序列数据分析与挖掘/ACTIVITY_1566153601293.csv", index_col=0, parse_dates=[0])
google_data = google_data.iloc[4:385]  #从6月份开始，此前数据时间间隔太大
# print(google_data.head())
'''
OUTPUT:
            lastSyncTime  steps  steps  runDistance  calories
date
2018-06-02    1527963944   2125      1445          105        46
2018-06-03    1528050378   1526      1033          162        38
2018-06-04    1528307835      0         0            0         0
2018-06-06    1528308561    211       143           16         5
2018-06-07    1528395102  10712      7909          359       268
'''

# 缺失数据Nan填充
google_data = google_data.fillna(method = 'ffill')
# print(google_data.iloc[1:50])

#日期缺失数据填充
# print(google_data.steps['20180602':'20180610'])
'''
OUTPUT:
date
2018-06-02     2125
2018-06-03     1526
2018-06-04        0
2018-06-06      211
2018-06-07    10712
2018-06-08     9876
2018-06-09     1033
2018-06-10     1824
Name: steps, dtype: int64
'''
google_data = google_data.resample('D').interpolate('linear')
# print(google_data.steps['20180602':'20180610'])
'''
OUtPUT:
date
2018-06-02     2125.0
2018-06-03     1526.0
2018-06-04        0.0
2018-06-05      105.5
2018-06-06      211.0
2018-06-07    10712.0
2018-06-08     9876.0
2018-06-09     1033.0
2018-06-10     1824.0
Freq: D, Name: steps, dtype: float64
'''

#--------时序可视化
# plt.plot(google_data['steps'])
# plt.title('Daily steps of activities')
# plt.show()


#--------平稳性检验
# diff = 0
# adf = ADF(google_data['steps'])
# if adf[1] > 0.05:
#     print(u'原始序列经检验不平稳，p值为：%s'%(adf[1]))
# else:
#     print(u'原始序列经检验平稳，p值为：%s'%(adf[1]))
'''
OUTPUT:
原始序列经检验平稳，p值为：8.06822189401253e-05
'''

# #--------时间序列预处理:纯随机性检验(白噪声检验)
# lb = acorr_ljungbox(google_data['steps'], lags=1)
# p = lb.values[0, 1]
# if p < 0.05:
# 	print(u'原始序列为非白噪声序列，p值为:%s' % p)
# else:
#     print(u'原始序列为白噪声序列，p值为:%s' % p)
'''
OUTPUT:
原始序列为非白噪声序列，p值为:5.045368315751089e-07
'''

#-------模型识别
# def plotds(xt, nlag=30, fig_size=(12, 10)):
#     #如果数据不是pd.Series则进行转换
#     if not isinstance(xt, pd.Series):
#         xt = pd.Series(xt)
#     plt.figure(figsize=fig_size)
#     layout = (2, 2)

#     #设置画图布局
#     ax_xt = plt.subplot2grid(layout, (0, 0), colspan=2)
#     ax_acf = plt.subplot2grid(layout, (1, 0))
#     ax_pacf = plt.subplot2grid(layout, (1, 1))
#     #开始画三类图：原始序列、acf、pacf
#     xt.plot(ax=ax_xt)
#     ax_xt.set_title('TIME SERIES')
#     plot_acf(xt, lags=nlag, ax=ax_acf)  #绘制自相关图
#     plot_pacf(xt, lags=nlag, ax=ax_pacf)   #绘制偏自相关图
#     plt.tight_layout()
#     plt.show()
#     return None

# plotds(google_data.steps.dropna(), nlag=50)
'''
通过查看ACF和PACF图，发现acf与pacf图像均呈现拖尾，
故应采用ARMA模型
'''

# # -------利用AIC准则进行模型定阶
data_df = google_data.copy()
aicVal = []
for ari in range(1, 3):
    for maj in range(0, 5):
        try:
            arima_obj = ARIMA(data_df.steps.tolist(), order=(ari, 0, maj))\
                .fit()  #最大似然估计法，可采用method='innovations_mle'
            aicVal.append([ari, maj, arima_obj.aic])
        except Exception as e:
            print(e)
print(aicVal)
'''
Output:
[[1, 0, 7266.595705707227], 
[1, 1, 7261.131826078026], 
[1, 2, 7260.8631700862], 
[1, 3, 7264.395895706072], 
[1, 4, 7265.659358199397], 
[2, 0, 7266.1385239626425], 
[2, 1, 7260.601091045484], 
[2, 2, 7262.885305505213], 
[2, 3, 7266.75290308656], 
[2, 4, 7267.189739774992]]
'''

#-------建立模型
#利用AIC准则，选择使AIC最小的阶数作为ARMA(p,q)的模型阶数，即p=2,q=1
# model = ARIMA(google_data.steps.tolist(), order=(2, 0, 1))
model = ARIMA(google_data.steps, order=(2, 0, 1))
arima_obj_fin = model.fit()
print(arima_obj_fin.summary())
'''
注意：
ARIMA(google_data.steps.tolist(), ...)时，model的_index属性为0,1,2,...形式
ARIMA(google_data.steps, ...)时，model的_index属性为'20121023'日期形式
'''
'''
Output:
                               SARIMAX Results
==============================================================================
Dep. Variable:                  steps   No. Observations:                  398
Model:                 ARIMA(2, 0, 1)   Log Likelihood               -3625.301
Date:                Wed, 20 Sep 2023   AIC                           7260.601
Time:                        13:59:46   BIC                           7280.533
Sample:                    06-02-2018   HQIC                          7268.496
                         - 07-04-2019
Covariance Type:                  opg
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const       3540.2163    291.644     12.139      0.000    2968.605    4111.828
ar.L1          1.0139      0.129      7.832      0.000       0.760       1.268
ar.L2         -0.1026      0.068     -1.510      0.131      -0.236       0.031
ma.L1         -0.8065      0.122     -6.610      0.000      -1.046      -0.567
sigma2      4.846e+06   2.63e+05     18.396      0.000    4.33e+06    5.36e+06
===================================================================================
Ljung-Box (L1) (Q):                   0.00   Jarque-Bera (JB):               337.09
Prob(Q):                              0.96   Prob(JB):                         0.00
Heteroskedasticity (H):               0.89   Skew:                             1.36
Prob(H) (two-sided):                  0.49   Kurtosis:                         6.60
===================================================================================
'''

# -------模型拟合与预测
#模型拟合:真实数据图像与拟合数据图像对比
# google_data['ARIMA'] = arima_obj_fin.predict()
# f, axarr = plt.subplots(1, sharex = True)
# f.set_size_inches(12, 8)
# google_data['steps'].iloc[len(google_data) - 100:]\
# .plot(color = 'b', linestyle = '-', ax = axarr)   #真实数据图像
# google_data['ARIMA'].iloc[len(google_data) - 100:]\
# .plot(color = 'r', linestyle = '--', ax = axarr)  #拟合数据图像
# axarr.set_title('ARIMA(2,0,1)')
# plt.xlabel('index')
# plt.ylabel('steps')
# plt.show()

# 对部分预测结果进行可视化展示
output1 = arima_obj_fin.predict('20190201', '20190207', dynamic=True, type='levels')  
steps_forcast = pd.concat([google_data.steps['20190115':'20190207'], output1], axis=1, keys=['original', 'predicted'])
plt.figure()
plt.plot(steps_forcast)
plt.title('Original vs predicted')
plt.show()


#预测后续数据
# 选择可视化的原始数据区间
true_data2 = google_data.steps['2019/06/17': '2019/07/04'] 
fig, ax = plt.subplots()
ax = true_data2.loc['2019/06/17': ].plot(ax=ax) # 设置坐标轴起点
plot_predict(arima_obj_fin, '2019/06/28', '2019/07/28', ax=ax) # 可视化预测时间起点与终点
plt.show()

#-------模型评估：平均绝对误差MAE，均方误差MSE，均方根误差RMSE
#对短期预测结果进行评估
short_label = google_data.steps['20190201': '20190202']
short_prediction = output1[:2]
short_mse_score = mse(short_label, short_prediction)
short_rmse_score = math.sqrt(short_mse_score)
short_mae_score = mae(short_label, short_prediction)
print('short_MSE: %.4f, short_RMSE: %.4f, short_MAE: %.4f' % (short_mse_score, short_rmse_score, short_mae_score))
#对长期预测结果进行评估
long_label = google_data.steps['20190201': '20190207']
long_prediction = output1
long_mse_score = mse(long_label, long_prediction)
long_rmse_score = math.sqrt(long_mse_score)
long_mae_score = mae(long_label, long_prediction)
print('long_MSE: %.4f, long_RMSE: %.4f, long_MAE: %.4f' % (long_mse_score, long_rmse_score, long_mae_score))
'''
OUTPUT:
short_MSE: 854620.4790, short_RMSE: 924.4569, short_MAE: 916.7704
long_MSE: 3390231.9682, long_RMSE: 1841.2583, long_MAE: 1571.2121
模型拟合与预测效果较差
'''