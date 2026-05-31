
'''
RMSnorm 的作用就是归一化,防止计算值过大或者过小导致数值不稳定,从而加速模型的训练和收敛.
它通过计算输入的均值和标准差来进行归一化,并且引入了一个可学习的缩放参数来调整归一化后的输出.
相比于 LayerNorm, RMSNorm 不需要计算均值,因此在某些情况下可以更高效地进行归一化.
通常防止梯度爆炸和梯度消失,从而提高模型的训练稳定性和性能.
'''
import torch

t = torch.rsqrt(torch.tensor(4.0))
print(t)

t2 = torch.ones(3,4)
print(t2)