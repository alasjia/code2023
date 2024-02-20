'''
作业4：LSTM-气温数据
'''


import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.utils.data as Data
import pandas as pd


#-----------数据读取与处理
#读取原始文件 并将日期设置为索引
data=pd.read_csv(r"E:\BJTU\其他\时间序列数据分析与挖掘\气温.csv",index_col="DATE",na_values="+9999,9")
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
index=pd.date_range(start=start_time,end=end_time,freq="H")
data=data.reindex(index)
#进行插值（部分补全的时间没有对应的数据）
data=data.interpolate()

# #可视化展示原始数据
# data.plot()
# plt.xlabel("Date")
# plt.ylabel("Teperature")
# plt.grid(True)
# plt.show()

#--------------设置超参数
input_size=1
hidden_size=128#256
output_size=1
epochs=100
lr=0.05
batch_size=20
time_step=12

#前140天用作训练
train_data=data[0:140*24]
#剩下的时间用作测试
test_data=data[140*24:]
##数据归一化
train_data_normalized=(train_data-train_data.min())/(train_data.max()-train_data.min())
test_data_normalized=(test_data-train_data.min())/(train_data.max()-train_data.min())                    

#---------------滑动窗口采样
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

#---------------构建LSTM网络
class MYLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,time_step):
        super(MYLSTM,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.time_step=time_step
        #创建LSTM层和linear层，LSTM层提取特征，linear层用作最后的预测
        self.lstm=nn.LSTM(
             input_size=self.input_size,
             hidden_size=self.hidden_size,
             num_layers=1,#2,
             batch_first=True,
             bidirectional=True
        )
        self.out=nn.Linear(self.hidden_size*2,self.output_size)
    def forward(self,x):
        #获得LSTM的计算结果，舍去h_n
        r_out,_=self.lstm(x)
        #按照LSTM模型结构修改input_seq的形状，作为linear层的输入
        r_out=r_out.reshape(-1,self.hidden_size*2)
        out=self.out(r_out)
        #将out恢复成（batch,seq_len,output_size
        out=out.reshape(-1,self.time_step,self.output_size)
        #return所有batch的seq_len的最后一项
        return out[:,-1,:]

#----------------模型参数初始化
#实例化神经网络
net=MYLSTM(input_size,hidden_size,output_size,time_step)
#初始化网络参数
for param in net.parameters():
    nn.init.normal_(param,mean=0,std=0.01)
#设置损失函数
loss=nn.MSELoss()
#设置优化器
optimizer=torch.optim.SGD(net.parameters(),lr=lr)
#如果GPU可用，就用GPU运算，否则使用CPU运算
device=torch.device("cuda:0"if torch.cuda.is_available()else"cpu")
#将net复制到device（GPU或CPU）
net.to(device)

#----------------模型训练
train_loss=[]
test_loss=[]
#模型训练
for epoch in range(epochs):#外循环训练一轮
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
        #反向传播
        l.backward()
        optimizer.step()
        train_1.append(l.item())
    #修改测试集的维度以便放入网络中
    test_x_temp=torch.unsqueeze(test_x,dim=2)
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
#绘制局部loss曲线
plt.plot(range(epochs),train_loss,label="train loss",linewidth=2)
plt.plot(range(epochs),test_loss,label="test loss",linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

#绘制局部loss曲线
plt.plot(range(epochs),train_loss,label="train loss",linewidth=2)
plt.plot(range(epochs),test_loss,label="test loss",linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.xlim(0,10)
plt.ylim(0,0.06)
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
#画出实际和预测的对比图
plt.plot(data.index[len(data)-len(test_y_result):],test_y_result,label='True')
plt.plot(data.index[len(data)-len(predict_result):],predict_result,label='Prediction')
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.show()
#与整体数据进行比较
plt.plot(data.index,data,label='True')
plt.plot(data.index[len(data)-len(predict_result):],predict_result,label='Prediction')
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.show()



'''
训练过程的loss值变化：
Epoch1:train loss=0.04957,test loss=0.16926
Epoch2:train loss=0.04347,test loss=0.15817
Epoch3:train loss=0.04206,test loss=0.16542
Epoch4:train loss=0.03968,test loss=0.14633
Epoch5:train loss=0.03559,test loss=0.12622
Epoch6:train loss=0.02866,test loss=0.08716
Epoch7:train loss=0.01945,test loss=0.05014
Epoch8:train loss=0.01054,test loss=0.02640
Epoch9:train loss=0.00514,test loss=0.00694
Epoch10:train loss=0.00325,test loss=0.00407
Epoch11:train loss=0.00289,test loss=0.00318
Epoch12:train loss=0.00278,test loss=0.00334
Epoch13:train loss=0.00276,test loss=0.00322
Epoch14:train loss=0.00271,test loss=0.00328
Epoch15:train loss=0.00269,test loss=0.00315
Epoch16:train loss=0.00267,test loss=0.00308
Epoch17:train loss=0.00264,test loss=0.00310
Epoch18:train loss=0.00263,test loss=0.00303
Epoch19:train loss=0.00262,test loss=0.00306
Epoch20:train loss=0.00258,test loss=0.00312
Epoch21:train loss=0.00255,test loss=0.00285
Epoch22:train loss=0.00253,test loss=0.00281
Epoch23:train loss=0.00252,test loss=0.00280
Epoch24:train loss=0.00250,test loss=0.00276
Epoch25:train loss=0.00247,test loss=0.00283
Epoch26:train loss=0.00244,test loss=0.00279
Epoch27:train loss=0.00242,test loss=0.00327
Epoch28:train loss=0.00242,test loss=0.00275
Epoch29:train loss=0.00238,test loss=0.00271
Epoch30:train loss=0.00236,test loss=0.00278
Epoch31:train loss=0.00235,test loss=0.00276
Epoch32:train loss=0.00233,test loss=0.00270
Epoch33:train loss=0.00231,test loss=0.00273
Epoch34:train loss=0.00229,test loss=0.00271
Epoch35:train loss=0.00228,test loss=0.00270
Epoch36:train loss=0.00225,test loss=0.00255
Epoch37:train loss=0.00225,test loss=0.00260
Epoch38:train loss=0.00223,test loss=0.00275
Epoch39:train loss=0.00222,test loss=0.00281
Epoch40:train loss=0.00222,test loss=0.00256
Epoch41:train loss=0.00218,test loss=0.00261
Epoch42:train loss=0.00218,test loss=0.00280
Epoch43:train loss=0.00217,test loss=0.00254
Epoch44:train loss=0.00215,test loss=0.00252
Epoch45:train loss=0.00213,test loss=0.00248
Epoch46:train loss=0.00213,test loss=0.00237
Epoch47:train loss=0.00210,test loss=0.00240
Epoch48:train loss=0.00209,test loss=0.00234
Epoch49:train loss=0.00209,test loss=0.00237
Epoch50:train loss=0.00207,test loss=0.00267
Epoch51:train loss=0.00206,test loss=0.00238
Epoch52:train loss=0.00205,test loss=0.00239
Epoch53:train loss=0.00204,test loss=0.00227
Epoch54:train loss=0.00202,test loss=0.00243
Epoch55:train loss=0.00202,test loss=0.00234
Epoch56:train loss=0.00201,test loss=0.00223
Epoch57:train loss=0.00199,test loss=0.00235
Epoch58:train loss=0.00198,test loss=0.00223
Epoch59:train loss=0.00197,test loss=0.00231
Epoch60:train loss=0.00198,test loss=0.00219
Epoch61:train loss=0.00195,test loss=0.00230
Epoch62:train loss=0.00194,test loss=0.00220
Epoch63:train loss=0.00193,test loss=0.00248
Epoch64:train loss=0.00192,test loss=0.00234
Epoch65:train loss=0.00191,test loss=0.00219
Epoch66:train loss=0.00189,test loss=0.00223
Epoch67:train loss=0.00189,test loss=0.00211
Epoch68:train loss=0.00189,test loss=0.00214
Epoch69:train loss=0.00188,test loss=0.00210
Epoch70:train loss=0.00186,test loss=0.00215
Epoch71:train loss=0.00186,test loss=0.00222
Epoch72:train loss=0.00187,test loss=0.00243
Epoch73:train loss=0.00184,test loss=0.00209
Epoch74:train loss=0.00184,test loss=0.00205
Epoch75:train loss=0.00182,test loss=0.00209
Epoch76:train loss=0.00181,test loss=0.00222
Epoch77:train loss=0.00181,test loss=0.00205
Epoch78:train loss=0.00180,test loss=0.00201
Epoch79:train loss=0.00179,test loss=0.00203
Epoch80:train loss=0.00178,test loss=0.00205
Epoch81:train loss=0.00177,test loss=0.00203
Epoch82:train loss=0.00177,test loss=0.00209
Epoch83:train loss=0.00176,test loss=0.00201
Epoch84:train loss=0.00175,test loss=0.00207
Epoch85:train loss=0.00176,test loss=0.00197
Epoch86:train loss=0.00175,test loss=0.00204
Epoch87:train loss=0.00174,test loss=0.00194
Epoch88:train loss=0.00172,test loss=0.00209
Epoch89:train loss=0.00172,test loss=0.00210
Epoch90:train loss=0.00172,test loss=0.00193
Epoch91:train loss=0.00171,test loss=0.00194
Epoch92:train loss=0.00170,test loss=0.00191
Epoch93:train loss=0.00169,test loss=0.00206
Epoch94:train loss=0.00169,test loss=0.00188
Epoch95:train loss=0.00169,test loss=0.00204
Epoch96:train loss=0.00168,test loss=0.00188
Epoch97:train loss=0.00167,test loss=0.00194
Epoch98:train loss=0.00168,test loss=0.00185
Epoch99:train loss=0.00166,test loss=0.00185
Epoch100:train loss=0.00165,test loss=0.00190
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
1.8388809
0.99285096
'''

'''
hidden_size = 256
1.7729683
0.97482014
'''

'''
num_layers = 2
3.6790783
1.4438319
'''