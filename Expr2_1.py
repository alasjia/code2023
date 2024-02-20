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


#---------数据清洗

#忽略警告输出
warnings.filterwarnings("ignore")
#用来正常显示中文标签
plt.rcParams["font.sans-serif"] = ["SimHei"]
#用来显示负号
plt.rcParams["axes.unicode_minus"] = False

#读取数据
path = "E:\\BJTU\\其他\\时间序列数据分析与挖掘\\作业及参考\\作业二\\实验2-SARIMA实验-数据\\Carbon_Dioxide_Emissions_From_Energy_Consumption-_Electric_Power_Sector.xlsx"
df = pd.read_excel(path, index_col = "Month")
#查看数据变量属性
# print(df.info())
#展示前三个数据
# print(df.head(3))
#显示尾部三个数据
# print(df.tail(3))
'''
- 列0：煤炭电力部门二氧化碳排放量，包含576个非空值，数据类型为float64。
- 列1：天然气电力部门二氧化碳排放量，包含576个非空值，数据类型为float64。
- 列2：蒸馏燃料，包括喷气式燃料、石油电力部门二氧化碳排放量，包含576个非空值，数据类型为float64。
- 列3：石油焦电力部门二氧化碳排放量，包含576个非空值，数据类型为float64。
- 列4：残留燃油电力部门二氧化碳排放量，包含576个非空值，数据类型为float64。
- 列5：石油电力部门二氧化碳排放量，包含576个非空值，数据类型为float64。
- 列6：地热能电力部门二氧化碳排放量，包含576个非空值，数据类型为object。
- 列7：非生物质废物电力部门二氧化碳排放量，包含576个非空值，数据类型为object。
- 列8：总能源电力部门二氧化碳排放量，包含576个非空值，数据类型为float64。
9 columns：其中6列和7列是object，其他是float
开始：1973-01-01 
结束：2020-12-01
'''

#去除空值
#用key——list 分割data
key_list = ["Geothermal Energy Electric Power Sector CO2 Emissions", "Non-Biomass Waste Electric Power Sector CO2 Emissions"]
df1 = df[df.keys().drop(key_list)]
# print(df1.head(3))
df2 = df[key_list]
# 删除值为Not Available的行
df2 = df2.drop(df2[df2.values == "Not Available"].index )
#将object转为float
df2 = df2.astype(np.float64)
# print(df2.head(3))
'''
现在有两个Dataframe：df1 和 df2
'''

#---------可视化处理
# #折线图
# #每种能源部的二氧化碳排放量大图
# keys1 = df1.keys() #提取行属性
# # print(keys1)
# fig, ax = plt.subplots(figsize=(28, 18))
# #
# # ax.plot(df1)
# # ax.plot(df2)
# df1.plot(ax = ax, fontsize=15)
# df2.plot(ax = ax, fontsize = 15)

# #设置
# ax.set_title("每种能源的碳排放", fontsize = 25)
# ax.set_xlabel("时间（月）", fontsize = 25)
# ax.set_ylabel("碳排放量（百万公吨", fontsize = 25)
# # ax.legend(loc = "best", fontsize = 18)
# ax.grid()
# plt.show()

# #每种能源的碳排子图
# #3*3子图
# fig, ax = plt.subplots(3, 3, figsize = (30, 18))
# axes = ax.flatten()
# fig.suptitle("每种能源的碳排放", fontsize = 25)

# #前7个
# keys1 = df1.keys()
# for key_index in range(len(keys1)):
#     key = keys1[key_index]
#     # axes[key_index].plot(df[key], label = key)
#     df[key].plot(ax = axes[key_index], fontsize = 10)

# #最后2个
# keys2 = df2.keys()
# for key_index in range(len(keys2)):
#     key = keys2[key_index]
#     axes_index = key_index + 7
#     df2[key].plot(ax = axes[axes_index], fontsize = 10)

# #设置子图
# for index in axes:
#     index.set_xlabel("时间（月）", fontsize = 20)
#     index.set_ylabel("碳排放量（百万公吨", fontsize = 20)
#     # index.legend(loc = "best", fontsize = 18)
#     index.grid()
#     plt.show()

# # 柱状图
# co2 = df2["1989-01-01":].sum()
# co2 = pd.concat([co2, df1.loc["1989-01-01":].sum()])
# co2 = co2.sort_values()
# print(co2)
# '''
# Geothermal Energy Electric Power Sector CO2 Emissions                                            12.417
# Distillate Fuel, Including Kerosene-Type Jet Fuel, Oil Electric Power Sector CO2 Emissions      224.773
# Non-Biomass Waste Electric Power Sector CO2 Emissions                                           328.774
# Petroleum Coke Electric Power Sector CO2 Emissions                                              366.577
# Residual Fuel Oil Electric Power Sector CO2 Emissions                                          1390.720
# Petroleum Electric Power Sector CO2 Emissions                                                  1982.074
# Natural Gas Electric Power Sector CO2 Emissions                                               11083.977
# Coal Electric Power Sector CO2 Emissions                                                      52428.046
# Total Energy Electric Power Sector CO2 Emissions                                              65835.286
# '''
# #每种能源的二氧化碳排放量柱状图
# cols = ["Geothermal Energy \n (地热)", 
#     "Non-Biomass Waste \n (非生物质废物)", 
#     "Petroleum Coke \n (石油焦)", 
#     "Distillate Fuel \n (蒸馏油)",
#     "Residual Fuel Oil \n (残余燃油)", 
#     "Petroleum \n (石油)", 
#     "Natural Gas \n (天然气)", 
#     "Coal \n (煤炭)", 
#     "Total Emissions \n (总碳排放)"]
# fig, ax = plt.subplots(figsize=(15, 10))#图的宽度和高度
# bar = ax.bar(cols, co2, align = "center", alpha = 0.8)    #"%%"就是字符格式的百分号
# ax.bar_label(bar, labels = ["%.2f%%"% (p*100) for p in co2/co2[-1]], fontsize = 20)
# ax.set_ylabel("碳排放量（百万公吨", fontsize = 20)
# ax.set_title("1989-2020年碳排放量数值比例图", fontsize = 25)
# ax.grid()
# plt.show()

# #画单个能源的图
# #使用天然气发电CO2碳排量
NGE = df1["Natural Gas Electric Power Sector CO2 Emissions"]
# print(NGE.head())
# '''
# Month
# 1973-01-01    12.175
# 1973-02-01    11.708
# 1973-03-01    13.994
# 1973-04-01    14.627
# 1973-05-01    17.344
# '''

# #天然气CO2排放量折线图
# fig, ax = plt.subplots(figsize = (15, 15))
# NGE.plot(ax = ax, fontsize = 15)
# ax.set_title("天然气碳排放", fontsize = 25)
# ax.set_xlabel("时间(月)", fontsize = 25)
# ax.set_ylabel("碳排放量(百万公吨)", fontsize = 25)
# ax.legend(loc = "best", fontsize = 15)   #loc = "best" 自动选择最佳图例放置位置
# ax.grid()
# plt.show()


# #-----------季节性和平稳性
# #利用STL工具分解时序
# decompostion = tsa.STL(NGE).fit()
# decompostion.plot()
# #趋势效应
# trend = decompostion.trend
# #季节效应
# seasonal = decompostion.seasonal
# #随机效应
# residual = decompostion.resid 
# plt.show()

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
test_stationarity(NGE)
'''
Test Statistic                   1.199387
p-value                          0.995982        
#Lags Used                      15.000000        
Number of Observations Used    560.000000        
Critical Value (1%)             -3.442081        
Critical Value (5%)             -2.866715        
Critical Value (10%)            -2.569526        
dtype: float64
X is not stationary
'''



#-----------序列平稳化以及非白噪音检验
def test_white_noise(data):
    return sm.stats.acorr_ljungbox(data.dropna(), return_df = True)
LjungBox_result = test_white_noise(NGE)
print(LjungBox_result)
'''
        lb_stat      lb_pvalue
1    510.761759  4.329901e-113
2    918.293107  3.937182e-200
3   1236.649128  8.193469e-268
4   1503.389860   0.000000e+00
5   1746.938516   0.000000e+00
6   1976.988610   0.000000e+00
7   2208.912086   0.000000e+00
8   2454.086461   0.000000e+00
9   2740.822623   0.000000e+00
10  3101.855589   0.000000e+00
可以看到在lag 1到lag 10的范围内，
所有的lb_pvalue都非常小（接近于0），
这意味着我们可以拒绝原假设，即时间序列数据存在自相关性。这表明该时间序列数据不是白噪声（white noise）
'''
#平稳性操作
#一阶差分
nge_diffence = NGE.diff(1)
#12步差分
nge_seasonal = nge_diffence.diff(12)
test_stationarity(nge_seasonal.dropna())
'''
Test Statistic                -9.874391e+00
p-value                        3.917473e-17
#Lags Used                     1.200000e+01
Number of Observations Used    5.500000e+02
Critical Value (1%)           -3.442296e+00
Critical Value (5%)           -2.866809e+00
Critical Value (10%)          -2.569576e+00
dtype: float64
X is stationary
'''

#----------建立SARIMA模型
# #图解法定阶
# fig = plot_acf(nge_seasonal.dropna(), lags = 40)
# fig = plot_pacf(nge_seasonal.dropna(), lags = 40)
# plt.show()

#网格搜索定阶
def grid_search(data):
    p = q = range(0,3)
    s = [12]
    d = [1]
    PDQs = list(itertools.product(p, d, q, s))
    pdq = list(itertools.product(p, d, q))
    params = []
    seasonal_params = []
    results = []
    grid = pd.DataFrame()

    for param in pdq:
        for seasonal_param in PDQs:
            mod = tsa.SARIMAX(data, order = param, seasonal_order = seasonal_param, enforce_stationarity = False, enforce_invertibility = False)
            result = mod.fit()
            print("ARIMA{}x{} - AIC:{}".format(param, seasonal_param, result.aic))
            params.append(param)
            seasonal_params.append(seasonal_param)
            results.append(result.aic)
    grid['pdq'] = params
    grid['PDQs'] = seasonal_params
    grid['aic'] = results
    print(grid[grid["aic"] == grid["aic"].min()])
# grid_search(NGE)
'''
          pdq           PDQs          aic
47  (1, 1, 2)  (0, 1, 2, 12)  2221.694862
最佳SARIMA模型参数为(1, 1, 2)和(0, 1, 2, 12)。
该模型的AIC值为2221.694862，AIC值越小表示模型拟合得越好。
参数中，(1, 1, 2)表示非季节性部分的ARIMA模型参数，(0, 1, 2, 12)表示季节性部分的ARIMA模型参数。
'''

#建立模型
mod = tsa.SARIMAX(NGE, order = (1,1,2), seasonal_order = (0,1,2,12))
results = mod.fit()
#检验模型参数部分是否满足非噪声条件
print(test_white_noise(results.resid))
fig_result = results.plot_diagnostics(figsize = (15,12))
plt.show()
'''
     lb_stat  lb_pvalue
1   0.183409   0.668460
2   0.235490   0.888923
3   1.269121   0.736478
4   1.395864   0.844914
5   1.397721   0.924562
6   1.414408   0.964976
7   2.795129   0.903286
8   4.053652   0.852251
9   4.656221   0.863188
10  7.237346   0.702864
p值均大于0.05，说明模型的残差部分为白噪声序列
'''

#----------模型预测
#获取预测结果，并评估预测误差
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

# #静态预测
# pred = get_prediction(NGE, results, '2015-01-01')
# plot_predition(pred)
'''
MSE: 5.458382432027038
RMSE: 2.3363181358768412
'''
# #动态预测
# pred = get_prediction(NGE, results, '2015-01-01', dynamic = True)
# plot_predition(pred)
'''
MSE: 41.22118547247422
RMSE: 6.420372689530898
'''
'''
静态预测的模型预测误差更小，相较于动态预测表现更好。这是因为动态预测在每个时间步都使用先前预测结果，导致误差逐渐累积。
虽然动态预测可能有更大的误差，但是其考虑了时间变化，因此能更好捕捉时间序列数据中的变化和趋势，从而更准确预测未来的值。
此外，动态预测具有更强适应性，可以根据实际观测值的变化来调整预测结果，这使得动态预测在处理非稳态时间序列数据时更具优势。
'''
#预测未来
def get_prediction_future(forecast, NGE):
    fig, ax = plt.subplots(figsize = (20, 16))
    #画出原始序列
    ax.plot(NGE, label = "Observe")
    #画出预测序列
    forecast.predicted_mean.plot(ax = ax, label = "Forecast")
    #画出置信区间
    ax.fill_between(forecast.conf_int().index, forecast.conf_int().iloc[:, 0], forecast.conf_int().iloc[:, 1], alpha = 0.4)
    ax.set_xlabel("时间（年）", fontsize = 18)
    ax.set_ylabel("天然气二氧化碳排放水平", fontsize = 18)
    ax.legend(loc = "best", fontsize = 18)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.show()
'''
conf_int()是一个用于获取时间序列模型预测结果的置信区间的方法。
'''
forecast = results.get_forecast(steps = 60)
get_prediction_future(forecast, NGE)

