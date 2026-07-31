import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
#-----------------------------------------------------
seq_len = 4  # 序列长度(每个序列有4个元素，1个元素是一年，12个月，即12个数据表示一个向量，构成一个元素)
vec_dim = 12  # 序列中每个元素的特征数目。本程序采用的序列元素为一年的旅客，一年12个月，即12维特征。

data = read_csv(r'./data/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=0)
data = np.array(data)  #(144, 1)
data2 = data[:,0]
sc = MinMaxScaler()
data = sc.fit_transform(data)  # 归一化
data = data.reshape(-1, vec_dim)  # torch.Size([12, 12])
train_x,train_y = [],[]

for i in range(data.shape[0] - seq_len):
    tmp_x=data[i:i+seq_len,:]#子序列  
    tmp_y=data[i+seq_len,:]#子序列后面的值  
    train_x.append(tmp_x)
    train_y.append(tmp_y)

train_x = torch.FloatTensor(train_x) #torch.Size([8, 4, 12]) torch.Size([8, 12])
train_y = torch.FloatTensor(train_y)
#------------------------------------------------------

class ManualGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ManualGRU, self).__init__()
        self.hidden_size = hidden_size

        # 重置门、更新门、候选隐状态的线性变换
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, init_state=None):
        # x shape: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.shape

        if init_state is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t = init_state

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]                     # (batch, input_size)
            combined = torch.cat((x_t, h_t), dim=1)  # (batch, input_size + hidden_size)

            r_t = torch.sigmoid(self.W_r(combined))   # 重置门
            z_t = torch.sigmoid(self.W_z(combined))   # 更新门

            # 候选隐状态，注意其输入用了 r_t * h_t 作为历史部分
            combined_h = torch.cat((x_t, r_t * h_t), dim=1)
            h_tilde = torch.tanh(self.W_h(combined_h))

            # 更新隐状态
            h_t = (1 - z_t) * h_t + z_t * h_tilde

            outputs.append(h_t)

        outputs = torch.stack(outputs, dim=1)    # (batch, seq_len, hidden_size)
        h_n = h_t.unsqueeze(0)                   # (1, batch, hidden_size)
        return outputs, h_n

class ManualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ManualLSTM, self).__init__()
        self.hidden_size = hidden_size
        # 四个门的线性变换：输入拼接后映射到 hidden_size
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, init_states=None):
        # x shape: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.shape

        # 初始化隐状态和细胞状态
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t, c_t = init_states

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]                     # (batch, input_size)
            combined = torch.cat((x_t, h_t), dim=1)   # (batch, input_size + hidden_size)

            i_t = torch.sigmoid(self.W_i(combined))
            f_t = torch.sigmoid(self.W_f(combined))
            o_t = torch.sigmoid(self.W_o(combined))
            c_tilde = torch.tanh(self.W_c(combined))

            c_t = f_t * c_t + i_t * c_tilde
            h_t = o_t * torch.tanh(c_t)

            outputs.append(h_t)

        # 将所有时间步的输出堆叠起来
        outputs = torch.stack(outputs, dim=1)    # (batch, seq_len, hidden_size)
        # 最后的隐状态和细胞状态，增加时间步维度 (1, batch, hidden_size)
        h_n = h_t.unsqueeze(0)
        c_n = c_t.unsqueeze(0)
        return outputs, (h_n, c_n)

class Air_Model(nn.Module):
    def __init__(self):
        super(Air_Model, self).__init__()
        #        输入x的维度12     隐含层h维度10（神经元个数）   隐含层的层数1 ?  默认batch_first=False,即batch在X的第2个维度
        # self.lstm=nn.LSTM(input_size=vec_dim, hidden_size=10, num_layers=1,batch_first=True, bidirectional=True, bias=True)
        # self.lstm = nn.GRU(input_size=vec_dim, hidden_size=10, num_layers=1, batch_first=True, bidirectional=False,bias=True)
        self.lstm = ManualGRU(input_size=vec_dim, hidden_size=10)
        self.linear = nn.Linear(10, vec_dim)

    def forward(self, x): #torch.Size([1, 4, 12])
        _, (h_out) = self.lstm(x)  # h_out是序列最后一个元素的hidden state
        # h_out = torch.mean(h_out, dim=0, keepdim=True)
        h_out = h_out.view(x.shape[0],-1)  # h_out's shape torch.Size([1, 10]) = (n_layer * n_direction, batchsize * hidden_dim), i.e. (1, 10)

        o = self.linear(h_out)
        return o


air_Model = Air_Model()
optimizer = torch.optim.Adam(air_Model.parameters(), lr=0.01)


for ep in range(400):
    for i, (x,y) in enumerate(zip(train_x,train_y)):
        x = x.unsqueeze(0) #加上批   torch.Size([1, 4, 12])
        pre_y = air_Model(x)  #torch.Size([1, 12])
        pre_y = torch.squeeze(pre_y) #torch.Size([12])
        loss = torch.nn.MSELoss()(pre_y, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 50 == 0:
            print('epoch:{:3d}, loss:{:6.4f}'.format(ep, loss.item()))


#---------------------------------------

torch.save(air_Model,'air_Model')

'''
'''
air_Model = torch.load('air_Model', weights_only=False)
air_Model.eval()
pre_data = []
for i, (x,y) in enumerate(zip(train_x,train_y)):
    x = x.unsqueeze(0) #加上批   torch.Size([1, 4, 12])
    pre_y = air_Model(x)  #torch.Size([1, 12])
    pre_data.append(pre_y.data.numpy())
    #print(pre_y.data.numpy())

#------------------------


plt.figure()
pre_data = np.array(pre_data)  #(8, 1, 12)
pre_data = pre_data.reshape(-1, 1).squeeze() #(8, 12) ---> (96,)

x_tick = np.arange(len(pre_data)) + (seq_len * vec_dim)
plt.plot(list(x_tick), pre_data, linewidth=2.5,   label='预测数据')  #从48开始
#------
ori_data = data.reshape(-1, 1).squeeze()  #(144,)

plt.plot(range(len(ori_data)), ori_data, linewidth=2.5,label='原始数据' ) #  据'

#plt.rcParams['font.sans-serif']=['SimHei']
plt.legend(fontsize=14)
plt.tick_params(labelsize=14)
plt.ylabel("数据的大小（已归一化）",fontsize=14) #Y轴标签

plt.xlabel("月份的序号",fontsize=14) #Y轴标签
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签simhei
plt.grid()
plt.show()


exit(0)
'''
'''
#绘制原始数据的曲线图=============================
plt.figure()


#------
ori_data = data.reshape(-1, 1).squeeze()  #(144,)

plt.plot(range(len(data2)), data2, linewidth=2.5  ) #  据'

#plt.rcParams['font.sans-serif']=['SimHei']

plt.tick_params(labelsize=14)
plt.ylabel("数据的大小",fontsize=14) #Y轴标签

plt.xlabel("月份的序号",fontsize=14) #Y轴标签
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签simhei
plt.grid()
plt.show()
