import torch
# x = torch.tensor([1,2,3,4,5])

# y= torch.tensor([10,20,30,40,50])

# condition = x>3
# result = torch.where(condition,x,y)#符合条件的元素取x,不符合条件的元素取y
# print(result)
# t = torch.arange(0, 10,2)
# t2 = torch.arange(5,1,-1)
# print(t)
# print(t2)

# v1 = torch.arange(1,4)
# v2 = torch.arange(1,3)
# v3=torch.outer(v1,v2)
# print(v3)

t1 = torch.tensor([[1, 2], 
                    [3, 4]])
t2 = torch.tensor([[5, 6], 
                    [7, 8]])

# dim=0 (沿着行拼接，即垂直堆叠)
# print(torch.cat([t1, t2], dim=0))
# # tensor([[1, 2],
# #         [3, 4],
# #         [5, 6],
# #         [7, 8]])
# # # 形状: (2, 2) 和 (2, 2) -> (4, 2)

# # dim=1 (沿着列拼接，即水平堆叠)
# print(torch.cat([t1, t2], dim=1))
# # tensor([[1, 2, 5, 6],
# #         [3, 4, 7, 8]])
# # # 形状: (2, 2) 和 (2, 2) -> (2, 4)

x = torch.randn(2, 4)
print(x.unsqueeze(-1).shape)
print(x.unsqueeze(-1))
print(x.unsqueeze(-2).shape)
print(x.unsqueeze(-2))
print(x.unsqueeze(-3).shape)
print(x.unsqueeze(-3))
# torch.Size([2, 4, 1])
# tensor([[[-0.7688],
#          [-0.5162],
#          [ 0.0212],
#          [ 0.9467]],

#         [[-0.9344],
#          [-0.9390],
#          [-1.9037],
#          [ 1.1769]]])
# torch.Size([2, 1, 4])
# tensor([[[-0.7688, -0.5162,  0.0212,  0.9467]],

#         [[-0.9344, -0.9390, -1.9037,  1.1769]]])
# torch.Size([1, 2, 4])
# tensor([[[-0.7688, -0.5162,  0.0212,  0.9467],
#          [-0.9344, -0.9390, -1.9037,  1.1769]]])

