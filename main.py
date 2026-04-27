# -*- coding: utf-8 -*-
"""
Created on Sun May 28 10:44:14 2023

@author: dell
"""

import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from torch.nn.utils import weight_norm
from sklearn.preprocessing import MinMaxScaler
from MLLA1D import MLLABlock
from ScaleGraphBlock import *
import random
import time  # ★ 新增：用于统计运行时间

# 记录程序开始时间
program_start_time = time.time()  # ★ 新增：从这里开始计时

# 解决画图中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 输入的历史look_back步，和预测未来的T步
look_back = 7
T = 1
epochs = 10  # 迭代次数
num_features = 8  # 输入特征数  由于MLLA中的k_max = feature_dim // (2 * len(channel_dims))限制,只能为双数
embed_dim = 128  # 嵌入维度 embed_dim must be divisible by num_heads 确保多头注意力机制能正确分割输入维度
dense_dim = 128  # 隐藏层神经元个数
num_heads = 8  # 头数
dropout_rate = 0.2  # 失活率
num_blocks = 3  # 编码器解码器数
learn_rate = 0.001  # 学习率
batch_size = 512  # 批大小 为2的次幂

# 固定随机种子
seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 读取数据 09081600 13313000 13310700 06221400
dataset = pd.read_excel('./实验数据/09081600.xlsx', usecols=[0,1,2,3,4,5,6,7])
dataX = dataset.values
dataY = dataset['X'].values

# 归一化数据
scaler1 = MinMaxScaler(feature_range=(0, 1))
scaler2 = MinMaxScaler(feature_range=(0, 1))
data_X = scaler1.fit_transform(dataX)
data_Y = scaler2.fit_transform(dataY.reshape(-1, 1))

# 划分训练集和测试集，用70%作为训练集，15%作为验证集，15%作为测试集
train_size = int(len(data_X) * 0.7)
val_size = int(len(data_X) * 0.15)
test_size = len(data_X) - train_size - val_size

train_X, train_Y = data_X[0:train_size], data_Y[0:train_size]
val_X, val_Y = data_X[train_size:train_size + val_size], data_Y[train_size:train_size + val_size]
test_X, test_Y = data_X[train_size + val_size:], data_Y[train_size + val_size:]


# 定义输入数据，输出标签数据的格式的函数，并将数据转换为模型可接受的3D格式
def create_dataset(datasetX, datasetY, look_back, T):
    dataX, dataY = [], []
    for i in range(0, len(datasetX) - look_back - T, T):
        a = datasetX[i:(i + look_back), :]
        dataX.append(a)
        if T == 1:
            dataY.append(datasetY[i + look_back])
        else:
            dataY.append(datasetY[i + look_back:i + look_back + T, 0])
    return np.array(dataX), np.array(dataY)


# 准备训练集和测试集的数据
trainX, trainY = create_dataset(train_X, train_Y, look_back, T)
valX, valY = create_dataset(val_X, val_Y, look_back, T)
val1X = np.concatenate((trainX, valX), axis=0)
val1Y = np.concatenate((trainY, valY), axis=0)
testX, testY = create_dataset(test_X, test_Y, look_back, T)

# 转换为PyTorch的Tensor数据
trainX = torch.Tensor(trainX)
trainY = torch.Tensor(trainY)
valX = torch.Tensor(valX)
valY = torch.Tensor(valY)
testX = torch.Tensor(testX)
testY = torch.Tensor(testY)
val1X = torch.Tensor(val1X)
val1Y = torch.Tensor(val1Y)


# 构建Transformer模型
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate):
        super(TransformerEncoder, self).__init__()

        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.dense1 = nn.Linear(embed_dim, dense_dim)
        self.dense2 = nn.Linear(dense_dim, embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        attn_output, _ = self.mha(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)

        dense_output = self.dense1(out1)
        dense_output = self.dense2(dense_output)
        dense_output = self.dropout2(dense_output)
        out2 = self.layernorm2(out1 + dense_output)

        return out2


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate):
        super(TransformerDecoder, self).__init__()

        self.mha1 = nn.MultiheadAttention(embed_dim, num_heads)
        self.mha2 = nn.MultiheadAttention(embed_dim, num_heads)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.layernorm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.dense1 = nn.Linear(embed_dim, dense_dim)
        self.dense2 = nn.Linear(dense_dim, embed_dim)
        self.layernorm4 = nn.LayerNorm(embed_dim)
        self.dropout4 = nn.Dropout(dropout_rate)

    def forward(self, inputs, encoder_outputs):
        attn1, _ = self.mha1(inputs, inputs, inputs)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(inputs + attn1)

        attn2, _ = self.mha2(out1, encoder_outputs, encoder_outputs)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(out1 + attn2)

        dense_output = self.dense1(out2)
        dense_output = self.dense2(dense_output)
        dense_output = self.dropout3(dense_output)
        out3 = self.layernorm3(out2 + dense_output)

        decoder_output = self.dense1(out3)
        decoder_output = self.dense2(decoder_output)
        decoder_output = self.dropout4(decoder_output)
        out4 = self.layernorm4(out3 + decoder_output)

        return out4


class Transformer(nn.Module):
    def __init__(self, num_features, embed_dim, dense_dim, num_heads, dropout_rate, num_blocks, output_sequence_length):
        super(Transformer, self).__init__()

        self.graph_block = GraphBlock(c_out=num_features, d_model=num_features, seq_len=look_back)
        self.attention_block = Attention_Block(d_model=num_features)
        self.mlla_block = MLLABlock(dim=num_features, input_resolution=look_back)
        self.embedding = nn.Linear(num_features, embed_dim)
        self.transformer_encoder = nn.ModuleList(
            [TransformerEncoder(embed_dim, dense_dim, num_heads, dropout_rate) for _ in range(num_blocks)])
        self.transformer_decoder = nn.ModuleList(
            [TransformerDecoder(embed_dim, dense_dim, num_heads, dropout_rate) for _ in range(num_blocks)])
        self.final_layer = nn.Linear(embed_dim * look_back, output_sequence_length)
        self.activation = nn.LeakyReLU()

    def forward(self, inputs):
        inputs = self.graph_block(inputs)
        inputs = self.attention_block(inputs)
        inputs = self.mlla_block(inputs)  # [batch, seq_len, num_features]
        encoder_inputs = inputs
        encoder_outputs = self.embedding(encoder_inputs)
        for i in range(len(self.transformer_encoder)):
            encoder_outputs = self.transformer_encoder[i](encoder_outputs)
        encoder_outputs = encoder_outputs.view(-1, encoder_outputs.shape[1] * encoder_outputs.shape[2])
        encoder_outputs = self.final_layer(encoder_outputs)
        # encoder_outputs = self.activation(encoder_outputs)
        decoder_outputs = encoder_outputs.view(-1, T)
        return decoder_outputs


# 定义训练集和测试集的数据加载器
class MyDataset(Dataset):
    def __init__(self, data_X, data_Y):
        self.data_X = data_X
        self.data_Y = data_Y

    def __getitem__(self, index):
        x = self.data_X[index]
        y = self.data_Y[index]
        return x, y

    def __len__(self):
        return len(self.data_X)


train_dataset = MyDataset(trainX, trainY)
val_dataset = MyDataset(valX, valY)
val1_dataset = MyDataset(val1X, val1Y)
test_dataset = MyDataset(testX, testY)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val1_loader = DataLoader(val1_dataset, batch_size=batch_size, shuffle=True)

# 创建模型实例
model = Transformer(num_features=num_features, embed_dim=embed_dim, dense_dim=dense_dim, num_heads=num_heads,
                    dropout_rate=dropout_rate, num_blocks=num_blocks, output_sequence_length=T)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ★ 新增：统计参数量（总参数和可训练参数）
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('=' * 50)
print(f'模型总参数量 (Total parameters): {total_params:,}')
print(f'可训练参数量 (Trainable parameters): {trainable_params:,}')
print('=' * 50)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

train_losses = []
val_losses = []
best_val_loss = float('inf')

# ★ 新增：统计训练时间
train_start_time = time.time()

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for inputs, labels in tqdm(train_loader, position=0):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val1_loader, position=0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val1_loader)
    val_losses.append(avg_val_loss)

    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # 保存验证损失最小的模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Best model saved at epoch {epoch + 1} with val loss {best_val_loss:.4f}")

# ★ 新增：训练时间统计结束
train_end_time = time.time()
train_time = train_end_time - train_start_time
print('\n' + '=' * 50)
print(f'训练总耗时 (Training time): {train_time:.2f} 秒，约 {train_time / 60:.2f} 分钟')
print(f'平均每个 epoch 耗时: {train_time / epochs:.2f} 秒')
print('=' * 50)

# 先加载验证集表现最好的模型
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

val_predictions = []
val_labels_list = []
with torch.no_grad():
    for inputs, labels in tqdm(val_loader, position=0):
        inputs = inputs.to(device)
        outputs = model(inputs)
        val_predictions.extend(outputs.cpu().numpy())
        val_labels_list.extend(labels.cpu().numpy())

# 可视化损失函数
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 单独绘制验证损失曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', color='#FF6F61', linewidth=2)
plt.title('Validation Loss Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

val_predictions = np.array(val_predictions).reshape(-1, 1)
val_labels = np.array(val_labels_list).reshape(-1, 1)

# 验证集数据反归一化
val_predictions = scaler2.inverse_transform(val_predictions)
val_labels = scaler2.inverse_transform(val_labels)


# 定义 nse 函数
def nse(observed, simulated):
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    nse_value = 1 - (numerator / denominator)
    return nse_value


# 计算验证集的评价指标（仅使用验证集数据）
val_r2 = r2_score(val_labels, val_predictions)
val_mae = mean_absolute_error(val_labels, val_predictions)
val_rmse = np.sqrt(mean_squared_error(val_labels, val_predictions))
val_mape = np.mean(np.abs((val_labels - val_predictions) / val_labels))
val_nse = nse(val_labels, val_predictions)

# 计算验证集的Kling-Gupta Efficiency (KGE)
val_obs_mean = np.mean(val_labels)
val_sim_mean = np.mean(val_predictions)
val_obs_std = np.std(val_labels)
val_sim_std = np.std(val_predictions)
val_r = np.corrcoef(val_labels.T, val_predictions.T)[0, 1]

val_beta = val_sim_mean / val_obs_mean
val_alpha = val_sim_std / val_obs_std

val_kge = 1 - np.sqrt((val_r - 1) ** 2 + (val_alpha - 1) ** 2 + (val_beta - 1) ** 2)

# 打印验证集的评价指标
print('Validation R2:', val_r2)
print('Validation MAE:', val_mae)
print('Validation RMSE:', val_rmse)
print('Validation MAPE:', val_mape)
print('Validation NSE:', val_nse)
print('Validation KGE:', val_kge)

# 可视化验证集结果与真实值的比较
plt.figure(figsize=(12, 6))
plt.plot(val_labels, label='真实值', linewidth=2)
plt.plot(val_predictions, label='预测值', linewidth=2, linestyle='--')
plt.xlabel('样本索引', fontsize=13)
plt.ylabel('数据值', fontsize=13)
plt.title('验证集结果与真实值比较', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 将验证集的结果保存为xlsx文件
val_temp = pd.DataFrame()
val_temp["实际值"] = val_labels.flatten()
val_temp["预测值"] = val_predictions.flatten()
val_temp.to_excel('验证集.xlsx', index=False)
print("验证集的预测结果已保存到 '验证集结果.xlsx' 文件中。")

# ===================== 测试阶段：统计推理时间 =====================
model.eval()
predictions = []

infer_start_time = time.time()  # ★ 新增：推理时间开始

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, position=0):
        inputs = inputs.to(device)
        outputs = model(inputs)
        predictions.extend(outputs.cpu().numpy())

infer_end_time = time.time()  # ★ 新增：推理时间结束

infer_time = infer_end_time - infer_start_time  # 整个测试集推理时间
num_test_samples = len(test_dataset)
time_per_sample = infer_time / num_test_samples if num_test_samples > 0 else float('nan')

print('\n' + '=' * 50)
print(f'测试集总推理时间 (Inference time on test set): {infer_time:.4f} 秒')
print(f'平均每个样本推理时间 (Avg. inference time per sample): {time_per_sample * 1000:.4f} 毫秒/样本')
print('=' * 50)

predictions = np.array(predictions).reshape(-1, 1)
labels = (testY.cpu().numpy()).reshape(-1, 1)

# 测试集数据反归一化
predictions = scaler2.inverse_transform(predictions)
labels = scaler2.inverse_transform(labels)

# 计算模型的评价指标
r2 = r2_score(labels, predictions)
mae = mean_absolute_error(labels, predictions)
rmse = np.sqrt(mean_squared_error(labels, predictions))
mape = np.mean(np.abs((labels - predictions) / labels))
nse_value = nse(labels, predictions)

# 计算Kling-Gupta Efficiency (KGE)
obs_mean = np.mean(labels)
sim_mean = np.mean(predictions)
obs_std = np.std(labels)
sim_std = np.std(predictions)
r = np.corrcoef(labels.T, predictions.T)[0, 1]

beta = sim_mean / obs_mean
alpha = sim_std / obs_std

kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

# 打印模型的评价指标
print('R2:', r2)
print('MAE:', mae)
print('RMSE:', rmse)
print('MAPE:', mape)
print('NSE:', nse_value)
print('KGE:', kge)

# 可视化预测结果
plt.xlabel('时间', fontsize=13)
plt.ylabel('数据', fontsize=13)
plt.plot(labels, label='真实值')
plt.plot(predictions, label='预测值')
plt.legend(fontsize=13)
plt.show()

# 将预测结果保存为xlsx文件
temp = pd.DataFrame()
temp["实际值"] = labels.flatten()
temp["预测值"] = predictions.flatten()
temp.to_excel('测试集.xlsx', index=False)
print("预测结果已保存到 '训练集结果.xlsl' 文件中。")

# ===================== 程序总耗时统计 =====================
program_end_time = time.time()
total_time = program_end_time - program_start_time

print('\n' + '=' * 50)
print(f'程序总耗时: {total_time:.2f} 秒，约 {total_time / 60:.2f} 分钟')
print('=' * 50)
