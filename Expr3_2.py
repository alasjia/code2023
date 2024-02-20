'''
作业2：CNN-真实数据回归-气温数据
'''

import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import mean_absolute_error as mae
torch.set_default_tensor_type(torch.DoubleTensor)


#绘制图像：时间轴和对应的数据列表作为输入，并根据给定的参数绘制序列数据图像。
def plot_series(time,series,format="-",start=0,end=None,label=None):
    #根据时间轴和对应数据列表绘制序列图像
    plt.plot(time[start:end],series[start:end],format,label=label)
    #设置横纵轴意义
    plt.xlabel("Time")
    plt.ylabel("Value")
    #设置图例说明字体大小
    if label:
        plt.legend(fontsize=14)
    #显示网格
    plt.grid(True)

#根据输入的比例将数据划分为训练集和测试集
def train_test_split(series,split_prop):
    train=series[:int(split_prop*int(series.size))]
    test=series[int(split_prop*int(series.size)):]
    return train,test

#滑窗、打乱
def data_process(train,test,window_size):
    #将训练集和测试集转换为张量（tensor）格式，并对训练集和测试集进行滑窗操作，得到短序列。
    train_tensor=torch.from_numpy(train)#训练集
    train_window_split=train_tensor.unfold(0,window_size,1)
    train_set=train_window_split.numpy()
    
    test_tensor=torch.from_numpy(test)#测试集
    test_window_split=test_tensor.unfold(0,window_size,1)
    test_set=test_window_split.numpy()
    #将训练集的短序列打乱顺序
    train_temp1=train_set.tolist()
    random.shuffle(train_temp1)
    train_temp2=np.array(train_temp1)
    
    test_temp1=test_set.tolist()
    test_temp2=np.array(test_temp1)
    #将以上的短序列分别划分为feature label
    train_feature_array=train_temp2[:,:window_size-1]
    train_label_array=train_temp2[:,window_size-1:]
    test_feature_array=test_temp2[:,:window_size-1]
    test_label_array=test_temp2[:,window_size-1:]
    #进一步将array形式转为tensor
    train_feature_tensor=torch.from_numpy(train_feature_array)
    train_label=torch.from_numpy(train_label_array)
    test_feature_tensor=torch.from_numpy(test_feature_array)
    test_label=torch.from_numpy(test_label_array)
    #扩展数据维度 符合CNN输入
    train_feature=train_feature_tensor.reshape(train_feature_tensor.shape[0],1,train_feature_tensor.shape[1])
    test_feature=test_feature_tensor.reshape(test_feature_tensor.shape[0],1,test_feature_tensor.shape[1])
    return train_feature,train_label,test_feature,test_label


#-------------数据读取及处理
#读取原始文件 并将日期设置为索引
data=pd.read_csv(r"E:\BJTU\其他\时间序列数据分析与挖掘\CNN实验数据\气温.csv",index_col="DATE",na_values="+9999,9")
#选取气温（TMP）一列
data=data["TMP"]
#将data.index设置为时间格式
data.index=pd.to_datetime(data.index)
#设置开始及阶数时间 选取一段数据
start_time=pd.to_datetime("2019-01-01 00:00:00")   
end_time=pd.to_datetime("2019-06-30 23:00:00")
data=data[start_time:end_time]
#从data中剔除TMP列为NaN的数据
data=data.dropna()
#将TMP数据从str转为int 并获取正确的数值
data=data.str.split(",",expand=True)[0]
data=data.astype("int")/10
#将时间补全（前面有丢弃操作）并将间隔设置为1h 重新设置data的索引     
time=pd.date_range(start=start_time,end=end_time,freq="H")
data=data.reindex(time)
#进行插值（部分补全的时间没有对应的数据）
data=data.interpolate()
#将数据转为array类型 与前面实验相同
series=np.array(data)


# #数据可视化
# fig,ax=plt.subplots(figsize=(20,6))
# #设置纵轴单位
# ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d℃'))
# plot_series(time,series,label='TMP')
# plt.show()

split_prop=0.7#设置划分比例
train,test=train_test_split(series,split_prop)#按照划分比例划分训练集、测试集
window_size=13#设置滑窗大小
#调用data_process进行数据处理
train_feature,train_label,test_feature,test_label=data_process(train,test,window_size)
#分别输出训练集、测试集的feature和label
# print(train_feature.shape)
# print(train_label.shape)
# print(test_feature.shape)
# print(test_label.shape)
'''
torch.Size([3028, 1, 12])
torch.Size([3028, 1])
torch.Size([1292, 1, 12])
torch.Size([1292, 1])
'''


#定义卷积神经网络
class ConvModulel(nn.Module):
    def __init__(self):
        super(ConvModulel,self).__init__()
        #两层一维卷积
        self.conv1=nn.Sequential(
             nn.Conv1d(in_channels=1,out_channels=32,kernel_size=3,stride=1,padding=1),
             nn.ReLU(inplace=True))
        self.conv2=nn.Sequential(
             nn.Conv1d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=1),
             nn.ReLU(inplace=True))
        #线形层
        self.fc1=nn.Linear(64,32)
        self.fc2=nn.Linear(32,1)

    def forward(self,X):
        out=self.conv1(X)#一维卷积
        out=F.max_pool1d(out,kernel_size=2,padding=1)#平均池化
        out=self.conv2(out)#一维卷积
        out=F.max_pool1d(out,3)#平均池化
        out=out.squeeze()
        out=self.fc1(out)
        out=self.fc2(out)
        #得到单值输出
        return out

#将数据划分为指定大小的batch
def data_iter(batch_size,features,labels):
    num_examples=len(features)
    indices=list(range(num_examples))
    for i in range(0,num_examples,batch_size):
        #最后一次可能不足一个batch
        j=torch.LongTensor(indices[i:min(i+batch_size,num_examples)])
        yield features.index_select(0,j),labels.index_select(0,j)

def square_loss(feature,label):
    return (net(feature)-label)**2/2
  
#预测函数 利用训练好的网格在测试集上进行测试、评估
def predict(x):
    temp=torch.from_numpy(x)
    x_tensor=temp.reshape(1,1,window_size-1)
    return net(x_tensor)


#--------------构建一维卷积神经网络
#构建网络
net=ConvModulel()
#参数初始化
for params in net.parameters():
    torch.nn.init.normal_(params,mean=0,std=0.01)
lr=0.001#学习率
num_epochs=100#训练轮数
batch_size=128#batch_size大小
loss=square_loss#损失函数
optimizer=torch.optim.Adam(net.parameters(),lr)#设置优化器


#-----------------训练模型
train_loss=[]
test_loss=[]
#模型训练
for epoch in range(num_epochs):#外循环训练一轮
    train_1,test_1=0.0,0.0
    for X,y in data_iter(batch_size,train_feature,train_label):
        #内循环训练一个batch
        l=loss(X,y).sum() #计算模型输出与真实数据之间的差距
        #梯度清零
        if optimizer is not None:
            optimizer.zero_grad()
        elif params is not None and params[0].gard is not None:
            for param in params:
                param.grad.data.zero_()
        #反向传播
        l.backward()
        optimizer.step()
    #该轮训练结束后 目前网络在训练集、测试集上的损失并输出
    train_1=loss((train_feature),train_label)
    test_1=loss((test_feature),test_label)
    train_loss.append(train_1.mean().item())
    test_loss.append(test_1.mean().item())
    print('epoch %d,train loss%f,test loss%f'%(epoch+1,train_1.mean().item(),test_1.mean().item()))
'''
epoch 1,train loss65.267920,test loss240.284693
epoch 2,train loss6.646067,test loss8.059949
epoch 3,train loss5.543915,test loss6.080081
epoch 4,train loss4.965376,test loss5.620263
epoch 5,train loss4.632377,test loss5.358809
epoch 6,train loss4.314488,test loss5.018347
epoch 7,train loss3.975276,test loss4.653992
epoch 8,train loss3.602532,test loss4.257821
epoch 9,train loss3.190245,test loss3.789017
epoch 10,train loss2.753982,test loss3.268397
epoch 11,train loss2.313384,test loss2.758670
epoch 12,train loss1.880064,test loss2.251190
epoch 13,train loss1.489187,test loss1.787898
epoch 14,train loss1.188353,test loss1.414124
epoch 15,train loss1.001883,test loss1.135171
epoch 16,train loss0.911090,test loss0.940493
epoch 17,train loss0.875921,test loss0.831587
epoch 18,train loss0.860702,test loss0.779817
epoch 19,train loss0.848401,test loss0.754855
epoch 20,train loss0.834597,test loss0.740064
epoch 21,train loss0.819554,test loss0.729821
epoch 22,train loss0.803749,test loss0.721891
epoch 23,train loss0.788688,test loss0.714646
epoch 24,train loss0.774739,test loss0.707976
epoch 25,train loss0.761153,test loss0.702423
epoch 26,train loss0.748678,test loss0.697216
epoch 27,train loss0.737362,test loss0.692453
epoch 28,train loss0.726771,test loss0.688295
epoch 29,train loss0.717091,test loss0.684697
epoch 30,train loss0.708435,test loss0.680870
epoch 31,train loss0.700487,test loss0.677311
epoch 32,train loss0.693204,test loss0.673916
epoch 33,train loss0.686489,test loss0.670578
epoch 34,train loss0.680262,test loss0.667392
epoch 35,train loss0.674500,test loss0.664363
epoch 36,train loss0.669108,test loss0.661286
epoch 37,train loss0.664051,test loss0.658294
epoch 38,train loss0.659235,test loss0.655183
epoch 39,train loss0.654533,test loss0.652606
epoch 40,train loss0.650092,test loss0.649638
epoch 41,train loss0.645806,test loss0.646949
epoch 42,train loss0.641713,test loss0.644027
epoch 43,train loss0.637746,test loss0.641013
epoch 44,train loss0.633983,test loss0.637476
epoch 45,train loss0.630298,test loss0.634305
epoch 46,train loss0.626753,test loss0.630383
epoch 47,train loss0.623313,test loss0.626421
epoch 48,train loss0.620016,test loss0.622262
epoch 49,train loss0.616815,test loss0.618399
epoch 50,train loss0.613748,test loss0.614932
epoch 51,train loss0.610564,test loss0.611863
epoch 52,train loss0.607580,test loss0.608538
epoch 53,train loss0.604564,test loss0.605301
epoch 54,train loss0.601721,test loss0.602514
epoch 55,train loss0.598836,test loss0.599898
epoch 56,train loss0.596184,test loss0.597154
epoch 57,train loss0.593561,test loss0.595044
epoch 58,train loss0.591035,test loss0.592319
epoch 59,train loss0.588436,test loss0.589578
epoch 60,train loss0.585917,test loss0.586483
epoch 61,train loss0.583286,test loss0.585263
epoch 62,train loss0.580650,test loss0.581337
epoch 63,train loss0.577862,test loss0.579784
epoch 64,train loss0.574549,test loss0.574957
epoch 65,train loss0.570320,test loss0.574427
epoch 66,train loss0.565182,test loss0.577369
epoch 67,train loss0.560085,test loss0.582533
epoch 68,train loss0.554156,test loss0.575920
epoch 69,train loss0.548304,test loss0.568063
epoch 70,train loss0.541718,test loss0.567328
epoch 71,train loss0.535116,test loss0.567924
epoch 72,train loss0.528508,test loss0.573560
epoch 73,train loss0.521318,test loss0.559967
epoch 74,train loss0.514520,test loss0.557240
epoch 75,train loss0.507968,test loss0.553242
epoch 76,train loss0.502150,test loss0.555263
epoch 77,train loss0.496561,test loss0.549636
epoch 78,train loss0.491094,test loss0.543010
epoch 79,train loss0.486802,test loss0.549904
epoch 80,train loss0.482546,test loss0.544614
epoch 81,train loss0.479047,test loss0.533428
epoch 82,train loss0.475197,test loss0.544574
epoch 83,train loss0.471874,test loss0.539271
epoch 84,train loss0.468691,test loss0.536773
epoch 85,train loss0.466033,test loss0.527147
epoch 86,train loss0.463901,test loss0.538669
epoch 87,train loss0.461315,test loss0.530211
epoch 88,train loss0.459254,test loss0.531750
epoch 89,train loss0.457278,test loss0.519357
epoch 90,train loss0.456468,test loss0.537948
epoch 91,train loss0.454247,test loss0.527269
epoch 92,train loss0.452761,test loss0.523793
epoch 93,train loss0.451316,test loss0.516326
epoch 94,train loss0.450920,test loss0.530074
epoch 95,train loss0.449442,test loss0.522167
epoch 96,train loss0.448353,test loss0.521767
epoch 97,train loss0.447498,test loss0.520323
epoch 98,train loss0.446868,test loss0.522393
epoch 99,train loss0.445842,test loss0.512511
epoch 100,train loss0.445725,test loss0.523387
'''


#--------------绘制损失函数loss曲线
#绘制loss曲线
x=np.arange(num_epochs)
plt.plot(x,train_loss,label="train_loss",linewidth=1.5)
plt.plot(x,test_loss,label="test_loss",linewidth=1.5)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

#绘制局部loss曲线
x=np.arange(num_epochs)
plt.plot(x,train_loss,label="train_loss",linewidth=1.5)
plt.plot(x,test_loss,label="test_loss",linewidth=1.5)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.xlim(20,num_epochs)
plt.ylim(0,1)
plt.show()


#---------------预测并对比
test_predict=[]
split_point=int(split_prop*int(series.size))
test_time=time[split_point+window_size-1:]
#测试集真实序列
test_true=series[split_point+window_size-1:]
#测试集预测序列
test_predict=net(test_feature).squeeze().tolist()
#将测试集真实数据与网格预测得到的数据以不同颜色画在一张图里 便于对比
plt.figure(figsize=(20,6))
plot_series(test_time,test_true,label='true')
plot_series(test_time,test_predict,label='predict')
plt.show()


mae_nn=mae(test_true,test_predict)
print(mae_nn)
'''
预测结果和真实结果之间的均方误差：
0.787310450499289
'''