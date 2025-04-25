import torch
from torch import nn

# # 定义矩阵维度 (示例使用2x3和3x2矩阵)
# A = torch.tensor([[1, 2, 3],
#                   [4, 5, 6]], dtype=torch.float32) # shape (2,3)
# B = torch.tensor([[7, 8],
#                   [9, 10],
#                   [11, 12]], dtype=torch.float32) # shape (3,2)
# bias = torch.tensor([[0.1, 0.2]], dtype=torch.float32) # shape (1,2) 行向量

# # 原始计算：A@B + bias（bias按行广播）
# result_original = A @ B + bias # shape (2,2)
# re = torch.addmm(bias,A,B)
# print("原始运算结果:\n", result_original,re)

# # 转置后的计算：B.T@A.T + bias.T（bias.T按列广播）
# result_transposed = B.T @ A.T + bias.T # shape (2,2)
# re_T = torch.addmm(bias.transpose(0,1),B.T,A.T)
# print("\n转置运算结果:\n", result_transposed,re_T)

# # 验证等式 (A@B + bias).T == B.T@A.T + bias.T
# print("\n是否互为转置:", torch.allclose(result_original.T, result_transposed, atol=1e-6))

# bsz = 2
# num_heads = 3
# q_seq_len = 4
# k_seq_len = 5

# # 原始偏置
# k_bias_T = torch.randn(bsz, q_seq_len, 1)  # [2,4,1]

# # 目标注意力权重形状
# attn_weights = torch.zeros(bsz, num_heads, q_seq_len, k_seq_len)

# # 执行扩展和加法
# k_bias_expanded = k_bias_T.unsqueeze(1).expand(-1, num_heads, -1, k_seq_len)

# attn_weights += k_bias_expanded

# # 验证形状和广播
# print(attn_weights.shape)  # torch.Size([2, 3, 4, 5])
# print((attn_weights[0, 0, :, 0] == k_bias_T[0, :, 0]).all())  # 应输出 True

# bias = nn.Parameter(torch.rand(3*4))

# # print(bias.shape)
# # bias = bias.unsqueeze(1)
# # print(bias.shape,bias.T.shape)
# a, b, c = bias.split(4, dim=0)
# print(bias)
# print(a, b, c)


# bsz, q_len, k_len, h = 2, 3, 4, 5

# # 初始化张量
# A = torch.randn(bsz, q_len, h)
# C = torch.randn(bsz, k_len, h)
# B = torch.randn(h, h)
# D = torch.randn(h, h)
# bias1 = torch.nn.Parameter(torch.zeros(h))
# bias2 = torch.nn.Parameter(torch.zeros(h))

# # (A*B + bias1)
# term1 = torch.matmul(A, B) + bias1  # [bsz, q_len, h]

# # (C*D + bias2)^T
# term2_T = torch.matmul(C, D).transpose(
#     1, 2) + bias2.unsqueeze(1)  # [bsz, h, k_len]

# # 原始结果
# result_original = torch.matmul(term1, term2_T)  # [bsz, q_len, k_len]


# # XWWX

# XW = torch.matmul(A, B) + bias1  # [bsz, q_len, h]

# XWW = torch.matmul(XW, D.transpose(0, 1))  # [bsz, q_len, h]

# biasT = bias2.unsqueeze(1)  # [h,1]

# biasT = torch.matmul(XW, biasT)  # [bsz, q_len, 1]

# XWWX = torch.matmul(XWW, C.transpose(-1, -2))  # [bsz, q_len, k_len]

# final = XWWX + biasT

# print("是否相等:", torch.allclose(result_original, final, atol=1e-6))


# A = torch.randn(bsz, h)
# print(A, A.shape)
# A = A.unsqueeze(1)
# print(A, A.shape)
# A = A.expand(-1, 2, -1)
# print(A, A.shape)


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def __repr__(self) -> str:
        return "Conv1D(nf={nf}, nx={nx})".format(**self.__dict__)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


input_dim = 5
output_dim = 3*input_dim

# 创建 Conv1D 层
conv1d_layer = Conv1D(nf=output_dim, nx=input_dim)

batch_size = 2
seq_len = 3
x = torch.randn(batch_size, seq_len, input_dim)

# w
print(conv1d_layer.weight,conv1d_layer.weight.shape)
w_q,w_k,w_v = conv1d_layer.weight.split(input_dim,dim=1)
print(w_q,w_q.shape)
print(w_k,w_k.shape)
print(w_v,w_v.shape)
q_b,k_b,v_b = conv1d_layer.bias.split(input_dim,dim=0)

q_=torch.addmm(q_b, x.view(-1, x.size(-1)), w_q)
q_=q_.view(batch_size, seq_len,input_dim)

# 使用 Conv1D 进行前向传播
output = conv1d_layer(x)  # 输出形状: (32, 50, input_dim24)

print(output,output.shape)

q,k,v = output.split(input_dim,dim=2)

print(q,q.shape)

print(q_,q_.shape)

# print(k,k.shape)
# print(v,v.shape)