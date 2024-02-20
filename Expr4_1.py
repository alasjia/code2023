'''
作业4：RNN-气温数据
'''


import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.utils.data as Data
import pandas as pd
from sklearn.metrics import mean_absolute_error as mae

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

#-----------生成混合模式的模拟时序数据
time=np.arange(4*365+1)   #时间数组
baseline=10    #基线值
slope=0.05     #斜率
amplitude=40   #振幅
noise_level=1   #噪音水平
series=baseline+trend(time,slope)\
       +seasonality(time,period=365,amplitude=amplitude)\
       +white_noise(time,noise_level=noise_level,seed=42)   #生成时间序列
# plt.figure(figsize=(10,6))
# plot_series(time,series)
# plt.show()  #显示


#--------------设置超参数
input_size=1
hidden_size=256
output_size=1
epochs=200
lr=0.05
batch_size=128
time_step=5

#---------------数据预处理
#训练集的比例
split_prop=0.7
#前70%的数据作为训练集
train_data=series[:int(split_prop*int(series.size))]
#剩下的数据作为测试集
test_data=series[int(split_prop*int(series.size)):]
#数据归一化：提升收敛速度和精度
train_data_normalized=(train_data-train_data.min())/(train_data.max()-train_data.min())
test_data_normalized=(test_data-train_data.min())/(train_data.max()-train_data.min())                      

#---------------滑动窗口采样
#设置list用于存储
train_x=[]
train_y=[]
test_x=[]
test_y=[]
#对训练数据采样
i=0
while (i+time_step+output_size<len(train_data_normalized)):
    #输入的序列
    train_x.append(train_data_normalized[i:i+time_step])
    #输出的序列
    train_y.append(train_data_normalized[i+time_step:i+time_step+output_size])
    i+=output_size
#对测试数据采样
j=0
while (j+time_step+output_size<len(test_data_normalized)):
    #输入的序列
    test_x.append(test_data_normalized[j:j+time_step])
    #输出的序列
    test_y.append(test_data_normalized[j+time_step:j+time_step+output_size])
    j+=output_size

#--------------装入数据
#将数据转换为tensor格式
train_x=torch.tensor(train_x,dtype=torch.float32)
train_y=torch.tensor(train_y,dtype=torch.float32)
test_x=torch.tensor(test_x,dtype=torch.float32)
test_y=torch.tensor(test_y,dtype=torch.float32)
#将训练数据装入DataLoader
train_dataset=Data.TensorDataset(train_x,train_y)
train_loader=Data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)

#---------------构建RNN网络
class MYRNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,time_step):
        super(MYRNN,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.time_step=time_step
        #创建RNN层和linear层，RNN层提取特征，linear层用作最后的预测
        self.rnn=nn.RNN(
             input_size=self.input_size,
             hidden_size=self.hidden_size,
             num_layers=1,
             batch_first=True,
        )
        self.out=nn.Linear(self.hidden_size,self.output_size)
    def forward(self,x):
        #获得RNN的计算结果，舍去h_n
        r_out,_=self.rnn(x)
        #按照RNN模型结构修改input_seq的形状，作为linear层的输入
        r_out=r_out.reshape(-1,self.hidden_size)
        out=self.out(r_out)
        #将out恢复成（batch,seq_len,output_size
        out=out.reshape(-1,self.time_step,self.output_size)
        #return所有batch的seq_len的最后一项
        return out[:,-1,:]

#----------------模型参数初始化
#实例化神经网络
net=MYRNN(input_size,hidden_size,output_size,time_step)
#初始化网络参数
for param in net.parameters():
    nn.init.normal_(param,mean=0,std=0.01)
#设置损失函数
loss=nn.MSELoss()
#设置优化器
optimizer=torch.optim.SGD(net.parameters(),lr=lr)
#如果GPU可用，就用GPU运算，否则使用CPU运算
device=torch.device("cuda:0"if torch.cuda.is_available() else"cpu")
#将net复制到device（GPU或CPU）
net.to(device)

#----------------模型训练
train_loss=[]
test_loss=[]
#开始训练
for epoch in range(epochs):
    train_1=[]
    test_1=0
    for x,y in train_loader:
        #RNN输入应为input(seq_len,batch,input_size),将x转化为三维数据
        x=torch.unsqueeze(x,dim=2)
        #将x，y放入device中
        x=x.to(device)
        y=y.to(device)
        #计算得到预测值y_predicet
        y_predict = net(x)
        #计算y_predicet与真实y的loss
        l=loss(y_predict,y)
        #清空所有被优化过的Variable的梯度
        optimizer.zero_grad()
        #反向传播，计算当前梯度
        l.backward()
        #根据梯度更新网络参数
        optimizer.step()
        train_1.append(l.item())
    #修改测试集的维度以便放入网络中
    test_x_temp=torch.unsqueeze(test_x,dim=2)
    #测试集放入device中
    test_x_temp=test_x_temp.to(device)
    test_y_temp=test_y.to(device)
    #得到测试集的预测结果
    test_predict=net(test_x_temp)
    #计算测试集loss
    test_1=loss(test_predict,test_y_temp)
    #打印
    print('Epoch%d:train loss=%.5f,test loss=%.5f'%(epoch+1,np.array(train_1).mean(),test_1.item()))
    train_loss.append(np.array(train_1).mean())
    test_loss.append(test_1.item())

#---------------loss与预测效果可视化展示
#绘制loss趋势图
plt.plot(range(epochs),train_loss,label="train loss",linewidth=2)
plt.plot(range(epochs),test_loss,label="test loss",linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

#绘制局部loss趋势图
plt.plot(range(epochs),train_loss,label="train loss",linewidth=2)
plt.plot(range(epochs),test_loss,label="test loss",linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.xlim(10,75)
plt.ylim(0,0.2)
plt.legend()
plt.show()

#将测试集放入模型计算预测结果
test_x_temp=torch.unsqueeze(test_x,dim=2)
test_x_temp=test_x_temp.to(device)
predict=net(test_x_temp)
#逆归一化
predict=predict.cpu().detach().numpy()*(train_data.max()-train_data.min())+train_data.min()
test_y=np.array(test_y)*(train_data.max()-train_data.min())+train_data.min()
#将数据从[[output_size],,,]转换为[x1,x2,x3]
predict_result=[]
test_y_result=[]
for item in predict:
    predict_result+=list(item)
for item in test_y:
    test_y_result+=list(item)

#指定figure的宽和高
fig_size=plt.rcParams['figure.figsize']
fig_size[0]=10
fig_size[1]=6
plt.rcParams['figure.figsize']=fig_size
# #画出实际和预测的对比图（局部）
# plt.plot(range(len(test_y_result)),test_y_result,label='True')
# plt.plot(range(len(predict_result)),predict_result,label='Prediction')
# plt.xlabel("Time")
# plt.ylabel("Value")
# plt.grid(True)
# plt.legend()
# plt.show()
# #与整体数据进行比较（全局）
# plt.plot(range(len(series)),series,label='True')
# plt.plot(range(len(series)-len(predict_result),len(series)),predict_result,label='Prediction')
# plt.xlabel("Time")
# plt.ylabel("Value")
# plt.grid(True)
# plt.legend()
# plt.show()

'''
Epoch1:train loss=0.19794,test loss=0.32196
Epoch2:train loss=0.05856,test loss=0.18383
Epoch3:train loss=0.03909,test loss=0.14565
Epoch4:train loss=0.03668,test loss=0.13332
Epoch5:train loss=0.03615,test loss=0.12698
Epoch6:train loss=0.03585,test loss=0.12397
Epoch7:train loss=0.03560,test loss=0.12144
Epoch8:train loss=0.03532,test loss=0.12087
Epoch9:train loss=0.03508,test loss=0.12092
Epoch10:train loss=0.03478,test loss=0.11859
Epoch11:train loss=0.03460,test loss=0.11947
Epoch12:train loss=0.03418,test loss=0.11808
Epoch13:train loss=0.03388,test loss=0.11771
Epoch14:train loss=0.03354,test loss=0.11696
Epoch15:train loss=0.03327,test loss=0.11574
Epoch16:train loss=0.03289,test loss=0.11246
Epoch17:train loss=0.03248,test loss=0.11100
Epoch18:train loss=0.03214,test loss=0.10874
Epoch19:train loss=0.03175,test loss=0.10757
Epoch20:train loss=0.03133,test loss=0.10712
Epoch21:train loss=0.03089,test loss=0.10494
Epoch22:train loss=0.03042,test loss=0.10384
Epoch23:train loss=0.02996,test loss=0.10272
Epoch24:train loss=0.02954,test loss=0.09888
Epoch25:train loss=0.02898,test loss=0.09686
Epoch26:train loss=0.02855,test loss=0.09566
Epoch27:train loss=0.02788,test loss=0.09368
Epoch28:train loss=0.02730,test loss=0.09134
Epoch29:train loss=0.02674,test loss=0.09050
Epoch30:train loss=0.02611,test loss=0.08679
Epoch31:train loss=0.02549,test loss=0.08440
Epoch32:train loss=0.02481,test loss=0.08458
Epoch33:train loss=0.02418,test loss=0.07841
Epoch34:train loss=0.02342,test loss=0.07724
Epoch35:train loss=0.02275,test loss=0.07215
Epoch36:train loss=0.02197,test loss=0.07088
Epoch37:train loss=0.02119,test loss=0.06829
Epoch38:train loss=0.02039,test loss=0.06601
Epoch39:train loss=0.01964,test loss=0.06228
Epoch40:train loss=0.01873,test loss=0.06090
Epoch41:train loss=0.01794,test loss=0.05849
Epoch42:train loss=0.01705,test loss=0.05392
Epoch43:train loss=0.01624,test loss=0.05148
Epoch44:train loss=0.01533,test loss=0.04896
Epoch45:train loss=0.01444,test loss=0.04526
Epoch46:train loss=0.01361,test loss=0.04322
Epoch47:train loss=0.01273,test loss=0.03950
Epoch48:train loss=0.01183,test loss=0.03592
Epoch49:train loss=0.01097,test loss=0.03230
Epoch50:train loss=0.01013,test loss=0.03069
Epoch51:train loss=0.00931,test loss=0.02944
Epoch52:train loss=0.00852,test loss=0.02541
Epoch53:train loss=0.00775,test loss=0.02329
Epoch54:train loss=0.00700,test loss=0.02128
Epoch55:train loss=0.00631,test loss=0.01843
Epoch56:train loss=0.00567,test loss=0.01559
Epoch57:train loss=0.00507,test loss=0.01398
Epoch58:train loss=0.00452,test loss=0.01273
Epoch59:train loss=0.00402,test loss=0.01098
Epoch60:train loss=0.00358,test loss=0.00912
Epoch61:train loss=0.00317,test loss=0.00776
Epoch62:train loss=0.00282,test loss=0.00663
Epoch63:train loss=0.00253,test loss=0.00543
Epoch64:train loss=0.00225,test loss=0.00491
Epoch65:train loss=0.00204,test loss=0.00410
Epoch66:train loss=0.00185,test loss=0.00374
Epoch67:train loss=0.00172,test loss=0.00294
Epoch68:train loss=0.00157,test loss=0.00292
Epoch69:train loss=0.00147,test loss=0.00272
Epoch70:train loss=0.00139,test loss=0.00221
Epoch71:train loss=0.00134,test loss=0.00179
Epoch72:train loss=0.00128,test loss=0.00176
Epoch73:train loss=0.00124,test loss=0.00157
Epoch74:train loss=0.00119,test loss=0.00149
Epoch75:train loss=0.00118,test loss=0.00147
Epoch76:train loss=0.00116,test loss=0.00134
Epoch77:train loss=0.00114,test loss=0.00138
Epoch78:train loss=0.00114,test loss=0.00123
Epoch79:train loss=0.00113,test loss=0.00115
Epoch80:train loss=0.00111,test loss=0.00124
Epoch81:train loss=0.00111,test loss=0.00118
Epoch82:train loss=0.00110,test loss=0.00112
Epoch83:train loss=0.00110,test loss=0.00108
Epoch84:train loss=0.00111,test loss=0.00107
Epoch85:train loss=0.00109,test loss=0.00112
Epoch86:train loss=0.00109,test loss=0.00107
Epoch87:train loss=0.00109,test loss=0.00108
Epoch88:train loss=0.00109,test loss=0.00108
Epoch89:train loss=0.00109,test loss=0.00107
Epoch90:train loss=0.00111,test loss=0.00103
Epoch91:train loss=0.00109,test loss=0.00106
Epoch92:train loss=0.00109,test loss=0.00105
Epoch93:train loss=0.00110,test loss=0.00105
Epoch94:train loss=0.00110,test loss=0.00103
Epoch95:train loss=0.00111,test loss=0.00102
Epoch96:train loss=0.00110,test loss=0.00103
Epoch97:train loss=0.00110,test loss=0.00105
Epoch98:train loss=0.00110,test loss=0.00103
Epoch99:train loss=0.00108,test loss=0.00107
Epoch100:train loss=0.00110,test loss=0.00101
Epoch101:train loss=0.00110,test loss=0.00103
Epoch102:train loss=0.00109,test loss=0.00102
Epoch103:train loss=0.00110,test loss=0.00104
Epoch104:train loss=0.00109,test loss=0.00105
Epoch105:train loss=0.00108,test loss=0.00109
Epoch106:train loss=0.00111,test loss=0.00102
Epoch107:train loss=0.00109,test loss=0.00104
Epoch108:train loss=0.00109,test loss=0.00105
Epoch109:train loss=0.00109,test loss=0.00105
Epoch110:train loss=0.00109,test loss=0.00105
Epoch111:train loss=0.00111,test loss=0.00102
Epoch112:train loss=0.00109,test loss=0.00104
Epoch113:train loss=0.00108,test loss=0.00106
Epoch114:train loss=0.00109,test loss=0.00104
Epoch115:train loss=0.00109,test loss=0.00104
Epoch116:train loss=0.00109,test loss=0.00106
Epoch117:train loss=0.00109,test loss=0.00103
Epoch118:train loss=0.00109,test loss=0.00107
Epoch119:train loss=0.00109,test loss=0.00103
Epoch120:train loss=0.00108,test loss=0.00103
Epoch121:train loss=0.00109,test loss=0.00105
Epoch122:train loss=0.00109,test loss=0.00102
Epoch123:train loss=0.00109,test loss=0.00103
Epoch124:train loss=0.00108,test loss=0.00105
Epoch125:train loss=0.00109,test loss=0.00104
Epoch126:train loss=0.00109,test loss=0.00101
Epoch127:train loss=0.00110,test loss=0.00102
Epoch128:train loss=0.00108,test loss=0.00106
Epoch129:train loss=0.00109,test loss=0.00103
Epoch130:train loss=0.00110,test loss=0.00101
Epoch131:train loss=0.00110,test loss=0.00101
Epoch132:train loss=0.00110,test loss=0.00102
Epoch133:train loss=0.00108,test loss=0.00104
Epoch134:train loss=0.00108,test loss=0.00105
Epoch135:train loss=0.00108,test loss=0.00104
Epoch136:train loss=0.00108,test loss=0.00104
Epoch137:train loss=0.00108,test loss=0.00103
Epoch138:train loss=0.00109,test loss=0.00101
Epoch139:train loss=0.00109,test loss=0.00105
Epoch140:train loss=0.00109,test loss=0.00101
Epoch141:train loss=0.00110,test loss=0.00100
Epoch142:train loss=0.00108,test loss=0.00107
Epoch143:train loss=0.00108,test loss=0.00104
Epoch144:train loss=0.00108,test loss=0.00106
Epoch145:train loss=0.00108,test loss=0.00104
Epoch146:train loss=0.00109,test loss=0.00103
Epoch147:train loss=0.00108,test loss=0.00101
Epoch148:train loss=0.00109,test loss=0.00101
Epoch149:train loss=0.00108,test loss=0.00102
Epoch150:train loss=0.00108,test loss=0.00106
Epoch151:train loss=0.00109,test loss=0.00103
Epoch152:train loss=0.00108,test loss=0.00107
Epoch153:train loss=0.00109,test loss=0.00102
Epoch154:train loss=0.00108,test loss=0.00103
Epoch155:train loss=0.00108,test loss=0.00101
Epoch156:train loss=0.00109,test loss=0.00104
Epoch157:train loss=0.00108,test loss=0.00103
Epoch158:train loss=0.00110,test loss=0.00102
Epoch159:train loss=0.00108,test loss=0.00102
Epoch160:train loss=0.00108,test loss=0.00104
Epoch161:train loss=0.00108,test loss=0.00103
Epoch162:train loss=0.00108,test loss=0.00103
Epoch163:train loss=0.00108,test loss=0.00105
Epoch164:train loss=0.00109,test loss=0.00104
Epoch165:train loss=0.00107,test loss=0.00105
Epoch166:train loss=0.00109,test loss=0.00102
Epoch167:train loss=0.00108,test loss=0.00101
Epoch168:train loss=0.00107,test loss=0.00105
Epoch169:train loss=0.00108,test loss=0.00105
Epoch170:train loss=0.00109,test loss=0.00102
Epoch171:train loss=0.00108,test loss=0.00101
Epoch172:train loss=0.00109,test loss=0.00103
Epoch173:train loss=0.00108,test loss=0.00103
Epoch174:train loss=0.00108,test loss=0.00101
Epoch175:train loss=0.00108,test loss=0.00103
Epoch176:train loss=0.00109,test loss=0.00101
Epoch177:train loss=0.00109,test loss=0.00102
Epoch178:train loss=0.00108,test loss=0.00103
Epoch179:train loss=0.00108,test loss=0.00102
Epoch180:train loss=0.00108,test loss=0.00101
Epoch181:train loss=0.00108,test loss=0.00103
Epoch182:train loss=0.00109,test loss=0.00100
Epoch183:train loss=0.00110,test loss=0.00100
Epoch184:train loss=0.00109,test loss=0.00105
Epoch185:train loss=0.00108,test loss=0.00102
Epoch186:train loss=0.00107,test loss=0.00105
Epoch187:train loss=0.00108,test loss=0.00102
Epoch188:train loss=0.00109,test loss=0.00101
Epoch189:train loss=0.00108,test loss=0.00103
Epoch190:train loss=0.00108,test loss=0.00103
Epoch191:train loss=0.00108,test loss=0.00104
Epoch192:train loss=0.00108,test loss=0.00101
Epoch193:train loss=0.00109,test loss=0.00102
Epoch194:train loss=0.00108,test loss=0.00103
Epoch195:train loss=0.00107,test loss=0.00103
Epoch196:train loss=0.00108,test loss=0.00103
Epoch197:train loss=0.00107,test loss=0.00105
Epoch198:train loss=0.00108,test loss=0.00103
Epoch199:train loss=0.00107,test loss=0.00102
Epoch200:train loss=0.00108,test loss=0.00101
'''

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
# 将预测结果和真实结果转换为numpy数组
predict_result = np.array(predict_result)
test_y_result = np.array(test_y_result)
# 计算均方误差和均方根误差
mse = mse(test_y_result, predict_result)
mae = mae(test_y_result, predict_result)
print(mse)
print(mae)
'''
10.697512
1.3445421
'''