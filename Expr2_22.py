import numpy as np 
import pandas as pd
import statsmodels.tsa.api as tsa
import statsmodels.api as sm 
import matplotlib.pyplot as plt 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import matplotlib


#---------数据准备

#忽略警告输出
warnings.filterwarnings("ignore")
#用来正常显示中文标签
plt.rcParams["font.sans-serif"] = ["SimHei"]
#用来显示负号
plt.rcParams["axes.unicode_minus"] = False

#读取数据
path = "E:\\BJTU\\其他\\时间序列数据分析与挖掘\\作业及参考\\作业二\\实验2-SARIMA实验-数据\\PeMS04_sensor94_flow_3h.csv"
tfflow = pd.read_csv(path, index_col = "time")
tfflow.reset_index(drop=True, inplace=True)
# 查看数据变量属性
print(tfflow.info())
# 展示前三个数据
print(tfflow.head(3))
# 显示尾部三个数据
print(tfflow.tail(3))
'''
<class 'pandas.core.frame.DataFrame'>
Index: 169 entries, 2018-01-20 00:00:00 to 2018-02-10 00:00:00
Data columns (total 1 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   flow    169 non-null    float64
dtypes: float64(1)
memory usage: 2.6+ KB
None
                       flow
time
2018-01-20 00:00:00  2423.0
2018-01-20 03:00:00  1796.0
2018-01-20 06:00:00  6738.0
                        flow
time
2018-02-09 18:00:00  14354.0
2018-02-09 21:00:00   7255.0
2018-02-10 00:00:00   2524.0
'''

# # ---------可视化处理
# fig, ax = plt.subplots(figsize=(15, 15))
# tfflow.plot(ax = ax, fontsize=15)
# #设置
# ax.set_title("Traffic Flow", fontsize = 25)
# ax.set_xlabel("")  
# ax.set_ylabel("flow", fontsize = 25)
# ax.legend(loc = "best", fontsize = 18)
# plt.xticks(rotation=15) #将x轴的label显示旋转15度
# ax.grid()
# plt.show()

#-----------检查季节特征和平稳特征

#利用ADF检查平稳性
def test_stationarity(timeseries, alpha = 1e-3):
    dftest = tsa.adfuller(timeseries, autolag = "AIC")
    dfoutput = pd.Series(dftest[0:4], index = ["Test Statistic", "p-value", "#Lags Used", "Number of Observations Used"])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)
    critical_value = dftest[4]["5%"]
    test_statistic = dftest[0]
    pvalue = dftest[1]
    if pvalue < alpha and test_statistic < critical_value:
        print("X is stationary")    
        return True
    else:
        print("X is not stationary")
        return False
# test_stationarity(tfflow)
'''
p-value                          0.152280
#Lags Used                       9.000000
Number of Observations Used    159.000000
Critical Value (1%)             -3.472161
Critical Value (5%)             -2.879895
Critical Value (10%)            -2.576557
dtype: float64
X is not stationary
'''

#差分法消除非平稳性
flow_seasonal = tfflow.diff(8)  
# test_stationarity(flow_seasonal.dropna())
'''
Test Statistic                -1.108411e+01
p-value                        4.239113e-20
#Lags Used                     1.000000e+00
Number of Observations Used    1.590000e+02
Critical Value (1%)           -3.472161e+00
Critical Value (5%)           -2.879895e+00
Critical Value (10%)          -2.576557e+00
dtype: float64
X is stationary
'''
#分别取差分阶数为1、8、56均可以获得平稳结果


# #利用STL工具分解时序
# flow_data = tfflow.flow
# decomposition = tsa.STL(flow_data, period=56).fit()  # 指定周期为56
# decomposition.plot()
# #趋势效应
# trend = decomposition.trend
# #季节效应
# seasonal = decomposition.seasonal
# #随机效应
# residual = decomposition.resid 
# plt.show()


#进行白噪声检验
def test_white_noise(data):
    return sm.stats.acorr_ljungbox(data.dropna(), return_df = True)
# LjungBox_result = test_white_noise(tfflow)
# LjungBox_result = test_white_noise(flow_seasonal)
# print(LjungBox_result)
'''
    原始数据白噪声检查结果：
        lb_stat      lb_pvalue
    1    54.997589   1.206777e-13
    2    56.819307   4.590334e-13
    3   109.649021   1.305774e-23
    4   201.875713   1.484504e-42
    5   253.614464   9.215647e-53
    6   254.770252   3.920744e-52
    7   306.918154   2.014008e-62
    8   451.065058   2.186417e-92
    9   499.745240  6.518872e-102
    10  501.304025  2.324052e-101
根据输出结果，可以看到lag 1到lag 10的p值远小于0.05，表明在该交通流时序数据中lag 1到lag 10都存在显著的自相关性，因此该数据不是白噪声。
# '''

#----------建立SARIMA模型
# #图解法定阶
# fig = plot_acf(tfflow.dropna(), lags = 80)
# fig = plot_pacf(tfflow.dropna(), lags = 80)
# plt.show()
'''
当输入原始数据时，ACF具有明显的局部周期特征，在56步时有突然增加的现象，同时ACF值保持较大说明其时间序列与其滞后版本之间存在较强的自相关关系；PACF图则具有较明显的拖尾特征
当输入经过季节差分处理后的flow_seasonal数据时，观察发现ACF和PACF图均呈现1阶拖尾，但由于看图判断有一定主观性，从而进一步采用AIC准则确定阶数
'''
'''
由于交通流数据显然以星期为周期单位，且时序数据以3小时为时间间隔，
因此分别设置季节可能的差分为8和56。
ARIMA部分的差分则尝试0和1。
接下来采用网格搜索法遍历可能的模型参数组合，依据AIC/BIC准则选择最佳的模型参数
'''
#网格搜索定阶
def grid_search(data):
    p = q = range(0,3)
    s = [8]
    d = range(0,3)
    PDQs = list(itertools.product(p, d, q, s))
    pdq = list(itertools.product(p, d, q))
    params = []
    seasonal_params = []
    aic_results = []
    bic_results = []
    grid = pd.DataFrame()
    for param in pdq:
        for seasonal_param in PDQs:
            mod = tsa.SARIMAX(data, order = param, seasonal_order = seasonal_param, enforce_stationarity = False, enforce_invertibility = False)
            result = mod.fit()
            print("ARIMA{}x{} - AIC:{}".format(param, seasonal_param, result.aic))
            params.append(param)
            seasonal_params.append(seasonal_param)
            aic_results.append(result.aic)
            bic_results.append(result.bic)
    grid['pdq'] = params
    grid['PDQs'] = seasonal_params
    grid['aic'] = aic_results
    grid['bic'] = bic_results
    print(grid[grid["aic"] == grid["aic"].min()])
    print(grid[grid["bic"] == grid["bic"].min()])
# grid_search(tfflow)

'''
bic和aic最小的参数组合均为SARIMA(0, 1, 2)x(2, 2, 2, 8) 
           pdq          PDQs          aic          bic
161  (0, 1, 2)  (2, 2, 2, 8)  2398.769975  2419.002419
           pdq          PDQs          aic          bic
161  (0, 1, 2)  (2, 2, 2, 8)  2398.769975  2419.002419
'''
'''
          pdq           PDQs        aic         bic
69  (2, 0, 1)  (2, 0, 0, 56)  857.15085  869.194849
          pdq           PDQs         aic         bic
60  (2, 0, 0)  (2, 0, 0, 56)  859.025198  869.061864
AIC最小的模型为  (2, 0, 1)x(2, 0, 0, 56) 
BIC最小的模型为  (2, 0, 0)x(2, 0, 0, 56) 
注意：(2, 0, 1)表示非季节性部分的ARIMA模型参数，(2, 0, 0, 56)表示季节性部分的ARIMA模型参数。
'''

#建立模型
mod = tsa.SARIMAX(tfflow, order = (0,1,2), seasonal_order = (2,2,2,8))
results = mod.fit()      
# print(results.summary())
#检验模型
# print(test_white_noise(results.resid))  #模型的残差部分     
'''
      lb_stat     lb_pvalue
1   15.638657  7.667109e-05
2   20.790486  3.057759e-05
3   22.302474  5.642888e-05
4   22.337961  1.716436e-04
5   27.383260  4.802948e-05
6   31.353024  2.170284e-05
7   38.229568  2.740358e-06
8   49.952831  4.172947e-08
9   53.694290  2.158012e-08
10  65.316903  3.525070e-10
模型残差部分为白噪声，说明已经充分提取其时序特征
'''
# # 获取诊断图
# fig_result = results.plot_diagnostics(figsize = (15,12))   
# plt.show()
'''
又诊断结果可以看出：
观察残差的时序图，可以看到残差基本稳定，随着时间的波动并没有很大的波动。
观察正态分布图和QQ图，模型的残差是基本服从正态分布的。
观察残差的自相关图，可以看到残差不存在自相关，说明残差序列是白噪声序列。
'''

#----------模型预测
#获取预测结果，并评估预测误差、
def get_prediction(data, results, start, dynamic = False):
    pred = results.get_prediction(start = start, dynamic = dynamic, full_results = True)
    pred_ci = pred.conf_int()
    forecast = pred.predicted_mean
    truth = data[start:]
    pred_concat = pd.concat([truth, forecast, pred_ci], axis = 1)
    pred_concat.columns = ['true', 'pred', 'up', 'low']
    print("MSE: {}".format(mean_squared_error(truth, forecast)))
    print("RMSE: {}".format(np.sqrt(mean_squared_error(truth, forecast))))
    return pred_concat

#绘制预测结果
def plot_predition(pred_concat):
    plt.fill_between(pred_concat.index, pred_concat['up'], pred_concat['low'], alpha = 0.4, label = 'pred_range')
    plt.plot(pred_concat['true'], label = 'true')
    plt.plot(pred_concat['pred'], label = 'pred')
    plt.legend()
    plt.show()

# # #静态预测
# pred = get_prediction(tfflow, results, 120)
# plot_predition(pred)
# '''
# MSE: 3291268.376836112
# RMSE: 1814.185320422396
# '''
# #动态预测
# pred = get_prediction(tfflow, results, 120, dynamic = True)
# plot_predition(pred)
# '''
# MSE: 25882210.967861637
# RMSE: 5087.456237439457
# '''

#预测未来
def get_prediction_future(forecast, tfflow):
    fig, ax = plt.subplots(figsize = (20, 16))
    #画出原始序列
    # ax.plot(tfflow, label = "Observe")
    #画出预测序列
    forecast.predicted_mean.plot(ax = ax, label = "Forecast")
    #画出置信区间
    # ax.fill_between(forecast.conf_int().index, forecast.conf_int().iloc[:, 0], forecast.conf_int().iloc[:, 1], alpha = 0.4)
    ax.set_xlabel("time", fontsize = 18)
    ax.set_ylabel("flow", fontsize = 18)
    ax.legend(loc = "best", fontsize = 18)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.show()
'''
conf_int()是一个用于获取时间序列模型预测结果的置信区间的方法。
'''
forecast = results.get_forecast(steps = 56)#预测未来7天
get_prediction_future(forecast, tfflow)
