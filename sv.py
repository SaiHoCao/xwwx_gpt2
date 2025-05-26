import torch

# ======================
# 参数设置
# ======================
torch.manual_seed(42)  # 固定随机种子
batch_size = 2         # 批大小
seq_len = 3            # 序列长度
dim = 4                # 特征维度

# ======================
# 生成模拟数据
# ======================
# 输入矩阵 [batch_size, seq_len, dim]
x = torch.randn(batch_size, seq_len, dim).round(decimals=2)
# 值投影矩阵 [dim, dim]
Wv = torch.randn(dim, dim).round(decimals=2)
# 注意力分数 [batch_size, seq_len, seq_len]
scores = torch.randn(batch_size, seq_len, seq_len).round(decimals=2)
scores = torch.softmax(scores, dim=-1)  # 转换为概率分布

# ======================
# 方法1：标准SV内积 (先投影后加权)
# ======================
# value = torch.einsum('bnd,dm->bnm', x, Wv)          # 投影得到V [batch, seq_len, dim]
value = torch.matmul(x, Wv)

# attn_standard = torch.einsum('bst,btd->bsd', scores, value)  # 加权求和
attn_standard = torch.matmul(scores, value)

# ======================
# 方法2：分解计算 (先加权后投影)
# ======================
# weighted_x = torch.einsum('bst,btd->bsd', scores, x)  # 加权原始输入 [batch, seq_len, dim]
weighted_x = torch.matmul(scores, x)
# attn_decomposed = torch.einsum('bsd,dm->bsm', weighted_x, Wv)  # 统一投影
attn_decomposed = torch.matmul(weighted_x, Wv)

# ======================
# 验证等价性
# ======================
print("结果一致性:", torch.allclose(attn_standard, attn_decomposed, atol=1e-6))
print("最大差异:", (attn_standard - attn_decomposed).abs().max().item())

# ======================
# 维度验证打印
# ======================
print("\n维度验证:")
print(f"原始输入维度: {x.shape}")
print(f"投影后Value维度: {value.shape}")
print(f"标准注意力输出维度: {attn_standard.shape}")
print(f"分解计算输出维度: {attn_decomposed.shape}")