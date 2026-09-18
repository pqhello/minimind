import torch
import torch.nn as nn

# dropout_layer = nn.Dropout(p=0.5)
# t1 = torch.tensor([1,2,3]).float()
# t2 = dropout_layer(t1)
# print(t2)
# #droupout丢弃之后为了保持期望不变，其他部分*（1/(1-p)）

# layer = nn.Linear(3, 5,bias=True)
# t1 = torch.tensor([1,2,3]).float()
# t2 = torch.tensor([[1,2,3]]).float()

# output1 = layer(t1)
# print(output1)
# output2 = layer(t2)
# print(output2)

#view和reshape的区别，view只能在内存连续的情况下使用，而reshape可以在内存不连续的情况下使用
# t1 = torch.tensor([[1,2,3],[4,5,6]])
# print(t1.view(-1,2))

#transpose和permute的区别，transpose只能交换两个维度，而permute可以交换任意多个维度,可以用于转置矩阵
# t1 = torch.tensor([[1,2,3],[4,5,6]])
# t2 = t1.transpose(0,1)
# print(t2)

# t1 = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
# print(torch.triu(t1,diagonal=1)) #上三角矩阵
# print(torch.triu(t1,diagonal=-1)) #下三角矩阵
# print(torch.triu(t1)) #转置矩阵

x = torch.tensor([0,1,1,3,2,1])
weights = torch.tensor([1.,1.,1.,2.,1.,1.])
count = torch.bincount(x,weights)
print(count)