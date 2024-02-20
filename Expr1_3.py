'''
该实验采用Kaggle数据集
完成任务二、四
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
steps：步数；连续变量；
steps：距离；连续变量；
runDistance：跑步距离；连续变量；
calories：卡路里；连续变量；
'''
kaggle_data = pd.read_csv("E:/BJTU/其他/时间序列数据分析与挖掘/Fitness tracker data (2016 - present) [2450+days]/01_Steps.csv",index_col=0, parse_dates=[0])
kaggle_data = kaggle_data.iloc[-1000:]  #取后1000天数据
# print(kaggle_data)
'''
           steps  distance  runDistance  calories
date
2020-04-20   8760      6021         1026       208
2020-04-21  10300      7744          356       269
2020-04-22   4908      3588           96       135
2020-04-23   5218      3459          309       122
2020-04-24   8387      5647          626       192
...           ...       ...          ...       ...
2023-01-10   4451      2919         2495       198
2023-01-11  11750      9207         7066       523
2023-01-12  11623      8484         6506       360
2023-01-13   4508      3169         2578       193
2023-01-14   2104      1408         1176       111

[1000 rows x 4 columns]
'''

# 缺失数据Nan填充
kaggle_data = kaggle_data.fillna(method = 'ffill')
# print(kaggle_data)

#日期缺失数据填充
# print(kaggle_data.steps['20200602':'20200620'])
kaggle_data = kaggle_data.resample('D').interpolate('linear')
# print(kaggle_data.steps['20200602':'20200620'])

# #--------时序可视化
# plt.plot(kaggle_data['steps'])
# plt.title('Daily steps of activities')
# plt.show()


# #--------平稳性检验
# diff = 0
# adf = ADF(kaggle_data['steps'])
# if adf[1] > 0.05:
#     print(u'原始序列经检验不平稳，p值为：%s'%(adf[1]))
# else:
#     print(u'原始序列经检验平稳，p值为：%s'%(adf[1]))
# '''
# OUTPUT:
# 原始序列经检验平稳，p值为：4.399024380870513e-26
# '''

# #--------时间序列预处理:纯随机性检验(白噪声检验)
# lb = acorr_ljungbox(kaggle_data['steps'], lags=1)
# p = lb.values[0, 1]
# if p < 0.05:
# 	print(u'原始序列为非白噪声序列，p值为:%s' % p)
# else:
#     print(u'原始序列为白噪声序列，p值为:%s' % p)
# '''
# OUTPUT:
# 原始序列为非白噪声序列，p值为:2.731034700481328e-20
# '''

# #-------观察一二阶差分图像
# #一阶差分处理
# kaggle_data['diff_1'] = kaggle_data['steps'].diff(1)
# #二阶差分
# kaggle_data['diff_2'] = kaggle_data['diff_1'].diff(1)
# print(kaggle_data.head())
# # '''
# #创建画板
# fig = plt.figure(1)
# #绘制原始图像
# ax1 = plt.subplot(3,1,1)
# plt.plot(kaggle_data['steps'])
# plt.title('raw data')
# #绘制一阶差分
# ax1 = plt.subplot(3,1,2)
# plt.plot(kaggle_data['diff_1'])
# plt.title('diff_1 data')
# #绘制二阶差分
# ax1 = plt.subplot(3,1,3)
# plt.plot(kaggle_data['diff_2'])
# plt.title('diff_2 data')
# #展示图像
# plt.show()

'''
观察发现结果符合预想场景，原数据本身为平稳数据，为对比不同模型结果，
接下来选择不同d参数进行建模
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

# plotds(kaggle_data.steps.dropna(), nlag=50)
'''
通过查看ACF和PACF图，发现acf与pacf图像均呈现拖尾，
故应采用ARMA模型
'''

# -------利用AIC准则进行模型定阶
data_df = kaggle_data.copy()
aicVal = []
for ari in range(1, 3):
    for maj in range(0, 5):
        try:
            arima_obj = ARIMA(data_df.steps.tolist(), order=(ari, 1, maj))\
                .fit()  #最大似然估计法，可采用method='innovations_mle'
            aicVal.append([ari, maj, arima_obj.aic])
        except Exception as e:
            print(e)
print(aicVal)
'''
Output:
[[1, 0, 20047.34427323275], 
[1, 1, 20036.32635415739], 
[1, 2, 20034.596542391006], 
[1, 3, 20035.155114995985], 
[1, 4, 20032.101019198322], 
[2, 0, 20041.770286924577], 
[2, 1, 20031.976899933146], 
[2, 2, 20029.650835986573], 
[2, 3, 20038.310421471393], 
[2, 4, 20039.12600800651]]
'''
#利用AIC准则，选择使AIC最小的三种（p,q)阶数作为ARIMA(p,d,q)的模型阶数，即(p=2,q=2)、(p=2,q=1)、(p=1,q=4)


#-------建立模型
def model_built(p,d,q):
    model = ARIMA(kaggle_data.steps.tolist(), order=(p, d, q))
    # model = ARIMA(kaggle_data.steps, order=(p, d, q))
    arima_obj_fin = model.fit()
    # print(arima_obj_fin.summary())
    return model

# -------模型拟合与预测
def model_predict(arima_model,p,d,q):
    arima_obj_fin = arima_model.fit()
    #模型拟合:真实数据图像与拟合数据图像对比
    kaggle_data['ARIMA'] = arima_obj_fin.predict()
    f, axarr = plt.subplots(1, sharex = True)
    f.set_size_inches(12, 8)
    kaggle_data['steps'].iloc[len(kaggle_data) - 100:]\
    .plot(color = 'b', linestyle = '-', ax = axarr)   #真实数据图像
    kaggle_data['ARIMA'].iloc[len(kaggle_data) - 100:]\
    .plot(color = 'r', linestyle = '--', ax = axarr)  #拟合数据图像
    axarr.set_title('ARIMA(%s,%s,%s)'%(p,d,q))
    plt.xlabel('index')
    plt.ylabel('steps')
    plt.show()

    # # 对部分预测结果进行可视化展示
    # output1 = arima_obj_fin.predict('2022/10/25', '2022/11/01', dynamic=True, type='levels')  
    # steps_forcast = pd.concat([kaggle_data.steps['2022/10/14': '2022/11/01'], output1], axis=1, keys=['original', 'predicted'])
    # plt.figure()
    # plt.plot(steps_forcast)
    # plt.title('Original vs predicted')
    # plt.show()

    #预测后续数据
    # 选择可视化的原始数据区间
    # true_data2 = kaggle_data.steps['2022/12/14': '2023/01/14'] 
    # fig, ax = plt.subplots()
    # ax = true_data2.loc['2022/12/14': ].plot(ax=ax) # 设置坐标轴起点
    # plot_predict(arima_obj_fin, '2023/01/01', '2023/02/01', ax=ax) # 可视化预测时间起点与终点
    # ax.set_title('ARIMA(%s,%s,%s)'%(p,d,q))
    # plt.show()

#-------模型评估：平均绝对误差MAE，均方误差MSE，均方根误差RMSE
def model_evaluate(arima_model):
    arima_obj_fin = arima_model.fit()
    output1 = arima_obj_fin.predict('2022/10/25', '2022/11/01', dynamic=True, type='levels')  
    #对短期预测结果进行评估
    short_label = kaggle_data.steps['2022/10/25':'2022/10/26']
    short_prediction = output1[:2]
    short_mse_score = mse(short_label, short_prediction)
    short_rmse_score = math.sqrt(short_mse_score)
    short_mae_score = mae(short_label, short_prediction)
    print('short_MSE: %.4f, short_RMSE: %.4f, short_MAE: %.4f' % (short_mse_score, short_rmse_score, short_mae_score))
    #对长期预测结果进行评估
    long_label = kaggle_data.steps['2022/10/25':'2022/11/01']
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
d = 0
# d = 1
p = 1
q = 0
# p = 2
# q = 1
# p = 1
# q = 4
model = model_built(p,d,q)
model_predict(model,p,d,q)
model_evaluate(model)
'''
2,0,2
short_MSE: 1825833.8174, short_RMSE: 1351.2342, short_MAE: 1295.4761
long_MSE: 46423909.3335, long_RMSE: 6813.5093, long_MAE: 5136.4491

2,1,2
short_MSE: 2490024.7198, short_RMSE: 1577.9812, short_MAE: 1517.9453
long_MSE: 43949411.0102, long_RMSE: 6629.4352, long_MAE: 5040.2703

'''