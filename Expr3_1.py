'''
作业1：CNN-模拟数据回归复现
'''


import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
torch.set_default_tensor_type(torch.DoubleTensor)

#绘制序列
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

#趋势模式
def trend(time,slope=0):
    #序列与时间呈线性关系
    return slope*time

#白噪声
def white_noise(time,noise_level=1,seed=None):
    #生成正太分布的伪随机数序列
    rnd=np.random.RandomState(seed)
    #noise_level控制噪声幅值大小
    return rnd.randn(len(time))*noise_level

#季节性（周期性）模式
def seasonal_pattern(season_time):
    """Just an arbitrary pattern,you can change it if you wish"""
    #分段函数（自变量取值[0,1]）作为一个模式
    return np.where(season_time<0.4,np.cos(season_time*2*np.pi),1/np.exp(3*season_time))

#将某个季节性（周期性）模式循环多次
def seasonality(time,period,amplitude=1,phase=0):
    """Repeats the same pattern at each period"""
    #将时间映射到0-1之间
    season_time=((time+phase)%period)/period
    return amplitude*seasonal_pattern(season_time)

#按照指定比例划分训练集和测试集
def train_test_split(series,split_prop):
    train=series[:int(split_prop*int(series.size))]
    test=series[int(split_prop*int(series.size)):]
    return train,test

#将数据划分为指定大小的batch
def data_iter(batch_size,features,labels):
    num_examples=len(features)
    indices=list(range(num_examples))
    for i in range(0,num_examples,batch_size):
        #最后一次可能不足一个batch
        j=torch.LongTensor(indices[i:min(i+batch_size,num_examples)])
        yield features.index_select(0,j),labels.index_select(0,j)

#滑窗、打乱
def data_process(train,test,window_size):#滑窗 打乱等数据处理
    #将数据转为tensor并进行滑窗 得到短序列
    train_tensor=torch.from_numpy(train)
    train_window_split=train_tensor.unfold(0,window_size,1)
    train_set=train_window_split.numpy()
    
    test_tensor=torch.from_numpy(test)
    test_window_split=test_tensor.unfold(0,window_size,1)
    test_set=test_window_split.numpy()
    #将训练集短序列打乱
    train_temp1=train_set.tolist()
    random.shuffle(train_temp1)
    train_temp2=np.array(train_temp1)

    #将短序列划分为feature label
    train_feature_array=train_temp2[:,:window_size-1]
    train_label_array=train_temp2[:,window_size-1:]
    test_feature_array=test_set[:,:window_size-1]
    test_label_array=test_set[:,window_size-1:]

    #将nadarray转为tensor
    train_feature_tensor=torch.from_numpy(train_feature_array)
    train_label=torch.from_numpy(train_label_array)
    test_feature_tensor=torch.from_numpy(test_feature_array)
    test_label=torch.from_numpy(test_label_array)

    #扩展数据维度 符合CNN输入（batch_size, channel, length）
    train_feature=train_feature_tensor.reshape(train_feature_tensor.shape[0],1,train_feature_tensor.shape[1])
    test_feature=test_feature_tensor.reshape(test_feature_tensor.shape[0],1,test_feature_tensor.shape[1])
    
    return train_feature,train_label,test_feature,test_label

#生成序列并绘制图像
time=np.arange(4*365+1)
baseline=10
slope=0.05
amplitude=40
noise_level=1
series=baseline+trend(time,slope)+seasonality(time,period=365,amplitude=amplitude)+white_noise(time,noise_level,seed=42)

# plt.figure(figsize=(10,6))
# plot_series(time,series)
# plt.show()

split_prop=0.7
train,test=train_test_split(series,split_prop)
window_size=6
train_feature,train_label,test_feature,test_label=data_process(train,test,window_size)
# print(train_feature.shape)
# print(train_label.shape)
# print(test_feature.shape)
# print(test_label.shape)
'''
torch.Size([1017, 1, 5])
torch.Size([1017, 1])
torch.Size([434, 1, 5])
torch.Size([434, 1])
'''


#定义卷积神经网络
class ConvModulel(nn.Module):
    def __init__(self):
        super(ConvModulel,self).__init__()
        #一层一维卷积
        self.conv1=nn.Sequential(
             nn.Conv1d(in_channels=1,out_channels=32,kernel_size=3,stride=1,padding=1),
             nn.ReLU(inplace=True))
        self.conv2=nn.Sequential(
             nn.Conv1d(in_channels=32,out_channels=64,kernel_size=2,stride=1,padding=1),
             nn.ReLU(inplace=True))
        #将输出通道变为单值
        self.fcl=nn.Linear(64,32)
        self.fc2=nn.Linear(32,1)
    def forward(self,X):
        out=self.conv1(X)#一维卷积
        out=F.avg_pool1d(out,2)#平均池化
        out=self.conv2(out)#一维卷积
        out=F.avg_pool1d(out,2)#平均池化
        out=out.squeeze()
        out=self.fcl(out)
        out=self.fc2(out)
        #out=self.fc3(out)
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


#定义模型
net=ConvModulel()
#参数初始化
for params in net.parameters():
    torch.nn.init.normal_(params,mean=0,std=0.01)
lr=0.005 #学习率
num_epochs=200 #训练轮数
batch_size=128 #batch_size大小
loss=square_loss #损失函数
optimizer=torch.optim.Adam(net.parameters(),lr) #设置优化器

train_loss=[]
test_loss=[]
#模型训练
for epoch in range(num_epochs):#外循环训练一轮
    train_1,test_1=0.0,0.0
    for X,y in data_iter(batch_size,train_feature,train_label):#内循环训练一个batch
        l=loss(X,y).sum()#计算模型输出与真实数据之间的差距
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
epoch 1,train loss1013.039721,test loss2870.418719
epoch 2,train loss37.287277,test loss71.351252
epoch 3,train loss65.232564,test loss174.745948
epoch 4,train loss13.989410,test loss19.547109
epoch 5,train loss12.781970,test loss13.245330
epoch 6,train loss14.432014,test loss22.270893
epoch 7,train loss12.886428,test loss13.651167
epoch 8,train loss12.305879,test loss15.090913
epoch 9,train loss11.971675,test loss13.393264
epoch 10,train loss11.853903,test loss13.282072
epoch 11,train loss11.750253,test loss14.103414
epoch 12,train loss11.716799,test loss13.134233
epoch 13,train loss11.360849,test loss13.715043
epoch 14,train loss11.216368,test loss13.735848
epoch 15,train loss11.075299,test loss12.767444
epoch 16,train loss10.819055,test loss12.545141
epoch 17,train loss10.594960,test loss13.221512
epoch 18,train loss10.375115,test loss12.865783
epoch 19,train loss10.171039,test loss12.285919
epoch 20,train loss10.015389,test loss11.853983
epoch 21,train loss9.883969,test loss11.623509
epoch 22,train loss9.759476,test loss11.548929
epoch 23,train loss9.650082,test loss11.617389
epoch 24,train loss9.577332,test loss11.841714
epoch 25,train loss9.549636,test loss12.223100
epoch 26,train loss9.562231,test loss12.735610
epoch 27,train loss9.607154,test loss13.322209
epoch 28,train loss9.668724,test loss13.917370
epoch 29,train loss9.728200,test loss14.476679
epoch 30,train loss9.775428,test loss15.001391
epoch 31,train loss9.768063,test loss15.482617
epoch 32,train loss9.425185,test loss14.960703
epoch 33,train loss9.834901,test loss17.048344
epoch 34,train loss8.614981,test loss13.194051
epoch 35,train loss8.191628,test loss11.992483
epoch 36,train loss8.040025,test loss12.059919
epoch 37,train loss7.833695,test loss11.953218
epoch 38,train loss7.246480,test loss10.057784
epoch 39,train loss7.119862,test loss9.977526
epoch 40,train loss7.006238,test loss10.094890
epoch 41,train loss6.723613,test loss9.435924
epoch 42,train loss6.462324,test loss8.574549
epoch 43,train loss6.430972,test loss8.879255
epoch 44,train loss6.092154,test loss7.724408
epoch 45,train loss5.952051,test loss7.207668
epoch 46,train loss5.892345,test loss7.008800
epoch 47,train loss5.769980,test loss7.289687
epoch 48,train loss5.690502,test loss7.209324
epoch 49,train loss5.609930,test loss6.988832
epoch 50,train loss5.554086,test loss7.090403
epoch 51,train loss5.489907,test loss6.954901
epoch 52,train loss5.438950,test loss6.909372
epoch 53,train loss5.396767,test loss6.913345
epoch 54,train loss5.356143,test loss6.865323
epoch 55,train loss5.321702,test loss6.845860
epoch 56,train loss5.293179,test loss6.841407
epoch 57,train loss5.268301,test loss6.825998
epoch 58,train loss5.247872,test loss6.825001
epoch 59,train loss5.230165,test loss6.816926
epoch 60,train loss5.214526,test loss6.807377
epoch 61,train loss5.197405,test loss6.794317
epoch 62,train loss5.182570,test loss6.778360
epoch 63,train loss5.169892,test loss6.762470
epoch 64,train loss5.158885,test loss6.757870
epoch 65,train loss5.148794,test loss6.752822
epoch 66,train loss5.139590,test loss6.744891
epoch 67,train loss5.131233,test loss6.738233
epoch 68,train loss5.123452,test loss6.733194
epoch 69,train loss5.115996,test loss6.728030
epoch 70,train loss5.109021,test loss6.722680
epoch 71,train loss5.102382,test loss6.715920
epoch 72,train loss5.096053,test loss6.709350
epoch 73,train loss5.090094,test loss6.704175
epoch 74,train loss5.084537,test loss6.700584
epoch 75,train loss5.079208,test loss6.695157
epoch 76,train loss5.074454,test loss6.687818
epoch 77,train loss5.070112,test loss6.685369
epoch 78,train loss5.065766,test loss6.688104
epoch 79,train loss5.061745,test loss6.687856
epoch 80,train loss5.058231,test loss6.685759
epoch 81,train loss5.054699,test loss6.683560
epoch 82,train loss5.051097,test loss6.684684
epoch 83,train loss5.047869,test loss6.686055
epoch 84,train loss5.044708,test loss6.684995
epoch 85,train loss5.041405,test loss6.682338
epoch 86,train loss5.038234,test loss6.679220
epoch 87,train loss5.035363,test loss6.674668
epoch 88,train loss5.032357,test loss6.670949
epoch 89,train loss5.029157,test loss6.667881
epoch 90,train loss5.026290,test loss6.664392
epoch 91,train loss5.023579,test loss6.659942
epoch 92,train loss5.020685,test loss6.655292
epoch 93,train loss5.017845,test loss6.650895
epoch 94,train loss5.015179,test loss6.646497
epoch 95,train loss5.012557,test loss6.641528
epoch 96,train loss5.009836,test loss6.636651
epoch 97,train loss5.007319,test loss6.632237
epoch 98,train loss5.004982,test loss6.628305
epoch 99,train loss5.002375,test loss6.624663
epoch 100,train loss4.999743,test loss6.620703
epoch 101,train loss4.997424,test loss6.616923
epoch 102,train loss4.995320,test loss6.612697
epoch 103,train loss4.992893,test loss6.608222
epoch 104,train loss4.990656,test loss6.603189
epoch 105,train loss4.989005,test loss6.598969
epoch 106,train loss4.987119,test loss6.594806
epoch 107,train loss4.984786,test loss6.590372
epoch 108,train loss4.982599,test loss6.586926
epoch 109,train loss4.980939,test loss6.583496
epoch 110,train loss4.979105,test loss6.579587
epoch 111,train loss4.977101,test loss6.575127
epoch 112,train loss4.975702,test loss6.570771
epoch 113,train loss4.974143,test loss6.566721
epoch 114,train loss4.971396,test loss6.562985
epoch 115,train loss4.969809,test loss6.559355
epoch 116,train loss4.968656,test loss6.555430
epoch 117,train loss4.966855,test loss6.551154
epoch 118,train loss4.966424,test loss6.548114
epoch 119,train loss4.964891,test loss6.545542
epoch 120,train loss4.963570,test loss6.542557
epoch 121,train loss4.960999,test loss6.539292
epoch 122,train loss4.959307,test loss6.536916
epoch 123,train loss4.957276,test loss6.534525
epoch 124,train loss4.956667,test loss6.531604
epoch 125,train loss4.954581,test loss6.528384
epoch 126,train loss4.954125,test loss6.525704
epoch 127,train loss4.953253,test loss6.523364
epoch 128,train loss4.951827,test loss6.520969
epoch 129,train loss4.949975,test loss6.518336
epoch 130,train loss4.948046,test loss6.515991
epoch 131,train loss4.946527,test loss6.513745
epoch 132,train loss4.944375,test loss6.511302
epoch 133,train loss4.942346,test loss6.508644
epoch 134,train loss4.940615,test loss6.506311
epoch 135,train loss4.940056,test loss6.504293
epoch 136,train loss4.938998,test loss6.502217
epoch 137,train loss4.938264,test loss6.500242
epoch 138,train loss4.937332,test loss6.498648
epoch 139,train loss4.937210,test loss6.497156
epoch 140,train loss4.936227,test loss6.495489
epoch 141,train loss4.935711,test loss6.493977
epoch 142,train loss4.934404,test loss6.492959
epoch 143,train loss4.934323,test loss6.491949
epoch 144,train loss4.933293,test loss6.490699
epoch 145,train loss4.932550,test loss6.489898
epoch 146,train loss4.932562,test loss6.489389
epoch 147,train loss4.932388,test loss6.488579
epoch 148,train loss4.931559,test loss6.488023
epoch 149,train loss4.932091,test loss6.488153
epoch 150,train loss4.932444,test loss6.487488
epoch 151,train loss4.932556,test loss6.486859
epoch 152,train loss4.930557,test loss6.485984
epoch 153,train loss4.928329,test loss6.484558
epoch 154,train loss4.927163,test loss6.483431
epoch 155,train loss4.927305,test loss6.482508
epoch 156,train loss4.925984,test loss6.481765
epoch 157,train loss4.924811,test loss6.481213
epoch 158,train loss4.925456,test loss6.481482
epoch 159,train loss4.926548,test loss6.482231
epoch 160,train loss4.927578,test loss6.482959
epoch 161,train loss4.928281,test loss6.483672
epoch 162,train loss4.927885,test loss6.484029
epoch 163,train loss4.926949,test loss6.484930
epoch 164,train loss4.924053,test loss6.483850
epoch 165,train loss4.937153,test loss6.497058
epoch 166,train loss4.988000,test loss6.566937
epoch 167,train loss4.987024,test loss6.567868
epoch 168,train loss4.961253,test loss6.529502
epoch 169,train loss4.935267,test loss6.494549
epoch 170,train loss4.919354,test loss6.476151
epoch 171,train loss4.910820,test loss6.468658
epoch 172,train loss4.907624,test loss6.466485
epoch 173,train loss4.908898,test loss6.467288
epoch 174,train loss4.911832,test loss6.469897
epoch 175,train loss4.913285,test loss6.473280
epoch 176,train loss4.916666,test loss6.477454
epoch 177,train loss4.920454,test loss6.481594
epoch 178,train loss4.924367,test loss6.485786
epoch 179,train loss4.928066,test loss6.490553
epoch 180,train loss4.930454,test loss6.494650
epoch 181,train loss4.930408,test loss6.496895
epoch 182,train loss4.928795,test loss6.496517
epoch 183,train loss4.926780,test loss6.495439
epoch 184,train loss4.924820,test loss6.494620
epoch 185,train loss4.924889,test loss6.495874
epoch 186,train loss4.927728,test loss6.498872
epoch 187,train loss4.928614,test loss6.501807
epoch 188,train loss4.929383,test loss6.503743
epoch 189,train loss4.927299,test loss6.502694
epoch 190,train loss4.922526,test loss6.501023
epoch 191,train loss4.918599,test loss6.498800
epoch 192,train loss4.915760,test loss6.497831
epoch 193,train loss4.914760,test loss6.499214
epoch 194,train loss4.914891,test loss6.501247
epoch 195,train loss4.915155,test loss6.502840
epoch 196,train loss4.915174,test loss6.504270
epoch 197,train loss4.915429,test loss6.505909
epoch 198,train loss4.909916,test loss6.503396
epoch 199,train loss4.906336,test loss6.503572
epoch 200,train loss4.907135,test loss6.506912
'''

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
plt.ylim(0,15)
plt.show()


#---------预测并对比
test_predict=[]
split_point=int(split_prop*int(series.size))
test_time=time[split_point+window_size-1:]
#测试集中真实数据
test_true=series[split_point+window_size-1:]
#测试集中预测数据
test_predict=net(test_feature).squeeze().tolist()
#依据测试集真实数据与grid预测数据，用不同颜色画在一张图里，便于对比
plt.figure(figsize=(10,6))
plot_series(test_time,test_true,label='true')
plot_series(test_time,test_predict,label='predict')
plt.show()

from sklearn.metrics import mean_absolute_error as mae
mae_nn=mae(test_true,test_predict)
print(mae_nn)
'''
预测结果和真实结果之间的均方误差：
1.410258171191532
'''