import torch
import torch.nn as nn

class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    """
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x

def verify_multihead_attention():
    # 设置随机种子确保结果可重现
    torch.manual_seed(42)
    
    # 定义多头注意力参数
    embed_dim = 12       # 嵌入维度
    num_heads = 3        # 头数
    head_dim = embed_dim // num_heads  # 每个头的维度
    
    # 创建QKV计算的Conv1D层
    conv1d_layer = Conv1D(nf=3 * embed_dim, nx=embed_dim)
    
    # 创建随机输入
    batch_size = 2
    seq_len = 4
    x = torch.randn(batch_size, seq_len, embed_dim)
    x_t = x.transpose(-1, -2)  # [batch_size, embed_dim, seq_len]
    
    print("=" * 50)
    print("多头注意力验证")
    print(f"嵌入维度: {embed_dim}, 头数: {num_heads}, 每头维度: {head_dim}")
    print("=" * 50)
    
    # 分割权重和偏置用于 Q, K, V 计算
    w_q, w_k, w_v = conv1d_layer.weight.split(embed_dim, dim=1)
    q_b, k_b, v_b = conv1d_layer.bias.split(embed_dim, dim=0)
    
    # 方法1: 使用整体权重直接计算，然后分割为多头
    output = conv1d_layer(x)  # [batch_size, seq_len, 3*embed_dim]
    q, k, v = output.split(embed_dim, dim=2)  # 每个 [batch_size, seq_len, embed_dim]
    
    # 重塑为多头形式
    # [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, num_heads, head_dim]
    q_multihead = q.view(batch_size, seq_len, num_heads, head_dim)
    k_multihead = k.view(batch_size, seq_len, num_heads, head_dim)
    v_multihead = v.view(batch_size, seq_len, num_heads, head_dim)
    
    # 交换维度以便进行注意力计算
    # [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
    q_multihead = q_multihead.transpose(1, 2)
    k_multihead = k_multihead.transpose(1, 2)
    v_multihead = v_multihead.transpose(1, 2)
    
    # 方法2: 分块计算，然后转换为多头
    # 这里使用 split 将权重进一步分割为每个头的权重
    w_q_heads = w_q.view(embed_dim, num_heads, head_dim)  # [embed_dim, num_heads, head_dim]
    w_k_heads = w_k.view(embed_dim, num_heads, head_dim)
    w_v_heads = w_v.view(embed_dim, num_heads, head_dim)
    
    q_b_heads = q_b.view(num_heads, head_dim)  # [num_heads, head_dim]
    k_b_heads = k_b.view(num_heads, head_dim)
    v_b_heads = v_b.view(num_heads, head_dim)
    
    # 使用逐头计算的方式直接得到多头格式
    q_split_multihead = []
    k_split_multihead = []
    v_split_multihead = []
    
    for h in range(num_heads):
        # 提取每个头的权重和偏置
        w_q_h = w_q_heads[:, h, :]  # [embed_dim, head_dim]
        w_k_h = w_k_heads[:, h, :]
        w_v_h = w_v_heads[:, h, :]
        
        q_b_h = q_b_heads[h]  # [head_dim]
        k_b_h = k_b_heads[h]
        v_b_h = v_b_heads[h]
        
        # 计算每个头的 Q, K, V
        q_h = torch.addmm(q_b_h, x.view(-1, embed_dim), w_q_h)
        q_h = q_h.view(batch_size, seq_len, head_dim)
        
        k_h = torch.addmm(k_b_h, x.view(-1, embed_dim), w_k_h)
        k_h = k_h.view(batch_size, seq_len, head_dim)
        
        v_h = torch.addmm(v_b_h, x.view(-1, embed_dim), w_v_h)
        v_h = v_h.view(batch_size, seq_len, head_dim)
        
        q_split_multihead.append(q_h)
        k_split_multihead.append(k_h)
        v_split_multihead.append(v_h)
    
    # 将列表转换为张量 [num_heads, batch_size, seq_len, head_dim]
    q_split_multihead = torch.stack(q_split_multihead, dim=0)
    k_split_multihead = torch.stack(k_split_multihead, dim=0)
    v_split_multihead = torch.stack(v_split_multihead, dim=0)
    
    # 调整维度顺序为 [batch_size, num_heads, seq_len, head_dim]
    q_split_multihead = q_split_multihead.transpose(0, 1)
    k_split_multihead = k_split_multihead.transpose(0, 1)
    v_split_multihead = v_split_multihead.transpose(0, 1)
    
    # 验证两种方法得到的多头Q、K、V是否一致
    print("验证多头Q、K、V是否一致:")
    print(f"Q 一致: {torch.allclose(q_multihead, q_split_multihead, rtol=1e-5)}")
    print(f"K 一致: {torch.allclose(k_multihead, k_split_multihead, rtol=1e-5)}")
    print(f"V 一致: {torch.allclose(v_multihead, v_split_multihead, rtol=1e-5)}")
    print("=" * 50)
    
    # 计算标准多头注意力分数
    # [batch_size, num_heads, seq_len, seq_len]
    attn_scores_standard = torch.matmul(q_multihead, k_multihead.transpose(-1, -2))
    
    
    # 方法2: 更进一步优化的计算方式 - 模拟 XWWX 计算,使用多批

    # step 1 Q*w_k^T Q*k_b^T
    q_k_t = torch.matmul(q,w_k.transpose(0,1))
    q_k_t = q_k_t.view(*q.shape[:-1],-1,head_dim).transpose(1,2)
    # [batch_size, num_heads, seq_len, head_dim]

    k_b = k_b.view(-1,head_dim)#[num_heads,head_dim] 
    k_b = k_b.unsqueeze(-1) #[num_heads,head_dim,1]
    k_bias_T = torch.matmul(q_multihead, k_b)  # [bsz,num_heads,q_seq_len,1]

    x_t_ = x_t.view(batch_size,-1,head_dim,seq_len)

    attn_weight = torch.matmul(q_k_t,x_t_) + k_bias_T

    # 修正的方法2: 使用多批处理的多头注意力

    # 1. 将q分割成多头形式
    q_heads = q.view(batch_size, seq_len, num_heads, head_dim)  # [batch_size, seq_len, num_heads, head_dim]
    q_heads = q_heads.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]

    # 2. 对w_k进行多头处理
    w_k_heads = w_k.view(embed_dim, num_heads, head_dim)  # [embed_dim, num_heads, head_dim]

    # 3. 计算Q·W_k^T
    q_w_k_t = torch.einsum('bnsh,enh->bnse', q_heads, w_k_heads)
    print(q_w_k_t.shape)
    # [batch_size, num_heads, seq_len, embed_dim]

    # 4. 计算Q·W_k^T·X^T
    x_t = x.transpose(-1, -2)  # [batch_size, embed_dim, seq_len]
    x_t_ = x_t.unsqueeze(1).expand(-1,num_heads,-1,-1)
    print(x_t_.shape)
    q_w_k_t_x_t = torch.matmul(q_w_k_t, x_t_)  # [batch_size, num_heads, seq_len, seq_len]

    # 5. 处理偏置 Q·k_b^T
    k_b_heads = k_b.view(num_heads, head_dim)  # [num_heads, head_dim]
    k_b_t_expanded = k_b_heads.unsqueeze(0).unsqueeze(2)  # [1, num_heads, 1, head_dim]
    q_k_b_t = torch.matmul(q_heads, k_b_t_expanded.transpose(-1, -2))  # [batch_size, num_heads, seq_len, 1]

    # 6. 扩展偏置项以匹配注意力分数的形状
    q_k_b_t = q_k_b_t.expand(-1, -1, -1, seq_len)  # [batch_size, num_heads, seq_len, seq_len]

    # 7. 计算完整的注意力分数 Q·(W_k^T·X^T) + Q·k_b^T
    attn_weight_fixed = q_w_k_t_x_t + q_k_b_t

    print(f"修复的多批方法与标准方法一致: {torch.allclose(attn_scores_standard, attn_weight_fixed, rtol=1e-5)}")

    # 方法三，使用循环处理多头
    attn_scores_xwwx = []
    
    for h in range(num_heads):
        # 获取每个头的权重
        w_q_h = w_q_heads[:, h, :]  # [embed_dim, head_dim]
        w_k_h = w_k_heads[:, h, :]
        
        q_b_h = q_b_heads[h]  # [head_dim]
        k_b_h = k_b_heads[h]

        # Step 1: 计算 Q = X·W_q + q_b
        q_h = torch.addmm(q_b_h, x.view(-1, embed_dim), w_q_h)
        q_h = q_h.view(batch_size, seq_len, head_dim)
        
        # 步骤1: 计算Q·W_k^T·X^T (不包含偏置)
        # q_h·W_k^T [batch_size, seq_len, embed_dim]
        xw_q_w_k_t = torch.matmul(q_h, w_k_h.t())
        
        # (X·W_q·W_k^T)·X^T [batch_size, seq_len, seq_len]
        xw_q_w_k_t_x_t = torch.matmul(xw_q_w_k_t, x_t)
        
        # 步骤2: 处理偏置项 - 计算完整表达式
        # (X·W_q + q_b)·(X·W_k + k_b)^T
        # = X·W_q·W_k^T·X^T + X·W_q·k_b^T + q_b·W_k^T·X^T + q_b·k_b^T
        
        # Q·k_b^T
        k_b_for_bcast = k_b_h.reshape(1, 1, -1)  # [1, 1, head_dim]
        xw_q_k_b_t = torch.matmul(q_h, k_b_for_bcast.transpose(-1, -2))
        
        
        # 组合所有部分
        qk_h_xwwx = xw_q_w_k_t_x_t + xw_q_k_b_t 
        
        attn_scores_xwwx.append(qk_h_xwwx)
    
    # 将列表转换为张量
    attn_scores_xwwx = torch.stack(attn_scores_xwwx, dim=0)
    # 调整维度顺序
    attn_scores_xwwx = attn_scores_xwwx.transpose(0, 1)
    
    print("XWWX优化计算验证:")
    print(f"XWWX计算QK^T 一致: {torch.allclose(attn_scores_standard, attn_weight, rtol=1e-5)}")
    print(f"XWWX2计算QK^T 一致: {torch.allclose(attn_scores_standard, attn_scores_xwwx, rtol=1e-5)}")
    print(f"差异最大值: {(attn_scores_standard - attn_scores_xwwx).abs().max()}")
    
    return {
        'q_multihead': q_multihead,
        'k_multihead': k_multihead,
        'v_multihead': v_multihead,
        'q_split_multihead': q_split_multihead,
        'k_split_multihead': k_split_multihead,
        'v_split_multihead': v_split_multihead,
        'attn_scores_standard': attn_scores_standard,
        'attn_scores_xwwx': attn_scores_xwwx,
        'attn_weight':attn_weight
    }

# 运行验证
results = verify_multihead_attention()