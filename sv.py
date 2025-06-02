import torch

# ======================
# 参数设置
# ======================
torch.manual_seed(42)
batch_size = 2
seq_len = 3
dim = 8    # 总特征维度
num_heads = 2
head_dim = dim // num_heads

# ======================
# 生成模拟数据 (带多头结构)
# ======================
x = torch.randn(batch_size, seq_len, dim).round(decimals=2)
Wv = torch.randn(dim, dim).round(decimals=2)  # 合并多头的大矩阵
v_bias = torch.randn(dim).round(decimals=2)   # 值偏置
scores = torch.randn(batch_size, num_heads, seq_len, seq_len).round(decimals=2)
scores = torch.softmax(scores, dim=-1)

# ======================
# 方法1：标准多头计算
# ======================
# 投影并分割多头
value = torch.matmul(x, Wv) + v_bias  # 加上偏置
value = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
# value: [batch, num_heads, seq_len, head_dim]
# scores: [batch, num_heads, seq_len, seq_len]
attn_standard = torch.matmul(scores, value)

# ======================
# 方法2：分解多头计算
# ======================
# 每个头独立加权
# scores: [batch, num_heads, seq_len, seq_len]
# x: [batch, seq_len, dim]
weighted_x = torch.einsum('bnst,btd->bnsd', scores, x)  # [2,3,2,4]

# 将偏置分成多头形式并扩展到正确的维度
v_bias_heads = v_bias.reshape(1, 1, num_heads, head_dim).expand(batch_size, seq_len, -1, -1).transpose(1, 2)
# v_bias_heads: [batch, num_heads, seq_len, head_dim]

# 统一投影并加上偏置
Wv_heads = Wv.view(dim, num_heads, head_dim)
# weighted_x: [batch, num_heads, seq_len, head_dim]
# Wv_heads: [dim, num_heads, head_dim]
attn_decomposed = torch.einsum('bnsd,dnh->bnsh', weighted_x, Wv_heads) + v_bias_heads

# ======================
# 验证等价性
# ======================
print(attn_standard)
print(attn_decomposed)
print("结果一致性:", torch.allclose(attn_standard, attn_decomposed, atol=1e-6))
print("最大差异:", (attn_standard - attn_decomposed).abs().max().item())

# ======================
# 维度验证打印
# ======================
print("\n关键张量维度:")
print(f"多头Value形状: {value.shape}")
print(f"标准输出合并前: {attn_standard.shape}")
print(f"分解加权中间结果: {weighted_x.shape}")
print(f"偏置多头形状: {v_bias_heads.shape}")