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

#进行白噪声检验
def test_white_noise(data):
    return sm.stats.acorr_ljungbox(data.dropna(), return_df = True)

#网格搜索定阶
def grid_search(data):
    p = q = range(0,3)
    s = [8]
    d = range(0,3)
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

#利用STL工具分解时序
def STL_decomposition(tfflow):
    flow_data = tfflow.flow
    decomposition = tsa.STL(flow_data, period=56).fit()  # 指定周期为56
    decomposition.plot()
    #趋势效应
    trend = decomposition.trend
    #季节效应
    seasonal = decomposition.seasonal
    #随机效应
    residual = decomposition.resid 
    plt.show()
    return 1

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

#获取预测未来数据
def get_prediction_future(forecast, tfflow):
    fig, ax = plt.subplots(figsize = (20, 16))
    #画出原始序列
    ax.plot(tfflow, label = "Observe")
    #画出预测序列
    forecast.predicted_mean.plot(ax = ax, label = "Forecast")
    #画出置信区间
    ax.fill_between(forecast.conf_int().index, forecast.conf_int().iloc[:, 0], forecast.conf_int().iloc[:, 1], alpha = 0.4)
    ax.set_xlabel("time", fontsize = 18)
    ax.set_ylabel("traffic flow", fontsize = 18)
    ax.legend(loc = "best", fontsize = 18)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.show()

if __name__=="__main__":
    #---------数据准备
    #忽略警告输出
    warnings.filterwarnings("ignore")
    #用来正常显示中文标签
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    #用来显示负号
    plt.rcParams["axes.unicode_minus"] = False

    #读取数据
    path = "E:\\BJTU\\其他\\时间序列数据分析与挖掘\\作业及参考\\作业二\\实验2-SARIMA实验-数据\\PeMS04_sensor7_flow_3h.csv"
    tfflow = pd.read_csv(path, index_col = "time")
    #重新设置索引格式，避免后续预测部分报错
    start_date = pd.to_datetime('2018-01-20 00:00:00')
    end_date = pd.to_datetime('2018-02-09 21:00:00')
    new_index = pd.date_range(start=start_date, end=end_date, freq='3H')
    tfflow.index = new_index
    # # 查看数据变量属性
    # print(tfflow.info())
    # # 展示前三个数据
    print(tfflow.head(3))
    # # 显示尾部三个数据
    # print(tfflow.tail(3))
    '''
    <class 'pandas.core.frame.DataFrame'>
    Index: 168 entries, 2018-01-20 00:00:00 to 2018-02-09 21:00:00
    Data columns (total 1 columns):
    #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
    0   flow    168 non-null    float64
    dtypes: float64(1)
    memory usage: 2.6+ KB
    None
                        flow
    time
    2018-01-20 00:00:00  1475.0
    2018-01-20 03:00:00   608.0
    2018-01-20 06:00:00  2266.0
                            flow
    time
    2018-02-09 15:00:00  12414.0
    2018-02-09 18:00:00   9121.0
    2018-02-09 21:00:00   4361.0
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
    '''
    观察发现交通流量数据虽然不是平稳的，但是具有很明显的季节特征
    '''


    #-----------检查季节特征和平稳特征
    # test_stationarity(tfflow)
    '''
    Test Statistic                  -3.368380
    p-value                          0.012085
    #Lags Used                      10.000000
    Number of Observations Used    157.000000
    Critical Value (1%)             -3.472703
    Critical Value (5%)             -2.880132
    Critical Value (10%)            -2.576683
    dtype: float64
    X is not stationary
    '''

    # #对季节特征进行差分
    # flow_seasonal1 = tfflow.diff(56)
    # flow_seasonal2 = tfflow.diff(8)
    # test_stationarity(flow_seasonal1.dropna())
    # test_stationarity(flow_seasonal2.dropna())
    '''
    由于交通流数据一般以星期或天为周期单位，且时序数据以3小时为时间间隔，
    因此尝试季节差分为7*（24/3） = 56步或者8步。
    56步平稳性检验结果：
    Test Statistic                -7.095540e+00
    p-value                        4.299000e-10
    #Lags Used                     0.000000e+00
    Number of Observations Used    1.110000e+02
    Critical Value (1%)           -3.490683e+00
    Critical Value (5%)           -2.887952e+00
    Critical Value (10%)          -2.580857e+00
    dtype: float64
    X is stationary
    8步平稳性检验结果：
    Test Statistic                  -4.666276
    p-value                          0.000097
    #Lags Used                      14.000000
    Number of Observations Used    145.000000
    Critical Value (1%)             -3.476273
    Critical Value (5%)             -2.881688
    Critical Value (10%)            -2.577513
    dtype: float64
    X is stationary

    根据ADF检验的结果，交通流时间序列经过8步或56步季节差分后均可平稳，
    因此可在该差分后序列上应用时间序列模型，接下来进一步尝试SARIMA建模和预测。
    '''

    # #白噪声检查
    # LjungBox_result = test_white_noise(tfflow)
    # print(LjungBox_result)
    '''
        lb_stat     lb_pvalue
    1    74.313854  6.663424e-18
    2    74.353106  7.152011e-17
    3   116.123652  5.274272e-25
    4   198.371997  8.411534e-42
    5   244.575778  8.013083e-51
    6   244.652921  5.693827e-50
    7   301.845788  2.440828e-61
    8   431.306126  3.734530e-88
    9   488.131511  1.997404e-99
    10  488.142593  1.507249e-98
    根据输出结果，可以看到lag 1到lag 10的p值远小于0.01，表明在该交通流时序数据中lag 1到lag 10都存在显著的自相关性，因此该数据不是白噪声。

    '''

    #----------建立SARIMA模型
    #图解法定阶
    # fig = plot_acf(tfflow.dropna(), lags = 80)
    # fig = plot_pacf(tfflow.dropna(), lags = 80)
    # fig = plot_acf(flow_seasonal1.dropna(), lags = 40)
    # fig = plot_pacf(flow_seasonal1.dropna(), lags = 40)
    # fig = plot_acf(flow_seasonal2.dropna(), lags = 40)
    # fig = plot_pacf(flow_seasonal2.dropna(), lags = 40)
    # plt.show()
    '''
    当输入原始数据时，ACF具有明显的局部周期特征，在56步时有突然增加的现象，同时ACF值保持较大说明其时间序列与其滞后版本之间存在较强的自相关关系；PACF图则具有较明显的拖尾特征
    当输入经过季节差分处理后的flow_seasonal1数据时，观察发现ACF和PACF图均呈现1阶拖尾，符合预期特征，从而可以进一步采用AIC准则确定阶数
    '''
    '''
    确定d与s取值后，采用网格搜索法遍历可能的pq组合，依据AIC准则选择最佳的模型参数
    '''
    #网格搜索法定阶
    grid_search(tfflow)
    '''
              pdq          PDQs          aic
    80  (2, 2, 2)  (2, 2, 2, 8)  2267.860345
    '''
    '''
            pdq           PDQs         aic
    14  (0, 0, 1)  (1, 0, 2, 56)  828.122692

    参数中，(0, 0, 1)表示非季节性部分的ARIMA模型参数，(1, 0, 2, 56)表示季节性部分的ARIMA模型参数。
    '''

    # 建立模型
    # mod = tsa.SARIMAX(tfflow, order = (0,0,1), seasonal_order = (1,0,2,56))   #s = 56
    mod = tsa.SARIMAX(tfflow, order = (2,2,2), seasonal_order = (2,2,2,8))    #s = 8
    results = mod.fit()
    # # 检验模型
    # print(test_white_noise(results.resid))  #模型的残差部分
    '''
    SARIMA(0, 0, 1)  (1, 0, 2, 56) :
            lb_stat     lb_pvalue
    1    93.776324  3.532723e-22
    2   129.986748  5.939314e-29
    3   139.610448  4.586204e-30
    4   142.453220  8.421193e-30
    5   148.764518  2.446197e-30
    6   172.920446  1.080066e-34
    7   240.957605  2.325152e-48
    8   353.337931  1.755368e-71
    9   420.330612  6.265344e-85
    10  445.275953  2.126076e-89
    SARIMA(2, 2, 2)  (2, 2, 2, 8)  :
          lb_stat  lb_pvalue
    1    0.289491   0.590547
    2    0.537177   0.764458
    3    1.190410   0.755305
    4    4.633802   0.326976
    5    6.671440   0.246245
    6    8.284062   0.218022
    7    8.940198   0.256978
    8   10.987416   0.202413
    9   11.167505   0.264402
    10  11.715010   0.304583

    依据“模型残差p值大于0.05即视为白噪声序列”的准则，当周期参数s=56时最优模型的残差非白噪声，存在显著的自相关性，说明模型没有完全捕捉到数据中的所有信息。
    当周期参数s=8时，最优模型的残差为白噪声，因此选择SARIMA(2, 2, 2)x(2, 2, 2, 8) 
    '''

    # # 画结果的诊断图
    # fig_result = results.plot_diagnostics(figsize = (15,12))
    # plt.show()

    # # ----------模型预测
    # #静态预测
    # pred = get_prediction(tfflow, results, 80)
    # plot_predition(pred) #绘制结果
    '''
    MSE: 1459226.0535683832
    RMSE: 1207.9842935934155
    '''

    # #动态预测
    # #由于动态预测具有时间累计误差，而模型拟合效果相较于复现实验具有较大差异，因此进行短期预测
    # pred = get_prediction(tfflow, results, 150, dynamic = True)
    # plot_predition(pred)
    '''
    MSE: 41751079.24203066
    RMSE: 6461.507505376022
    '''

    '''
    conf_int()是一个用于获取时间序列模型预测结果的置信区间的方法。
    '''
    forecast = results.get_forecast(steps = 56)  #未来7天=8*7=56步
    #预测未来
    get_prediction_future(forecast, tfflow)
