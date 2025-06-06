import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class GPT2PerformanceAnalyzer:
    def __init__(self, d_model=1024, n_heads=16, d_ff=4096, vocab_size=50257, n_layers=24,fp16=False):
        """
        GPT-2 性能分析工具
        
        参数:
        d_model: 模型隐藏层维度 (默认: 1024)
        n_heads: 注意力头数 (默认: 16)
        d_ff: FFN层中间维度 (默认: 4096)
        vocab_size: 词表大小 (默认: 50257)
        n_layers: 层数 (默认: 24)
        fp16: 是否使用FP16精度 (默认: False)
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.bytes_per_value = 2 if fp16 else 4  # FP16精度，2字节/值，FP32精度，4字节/值
        self.n_layers = n_layers
        self.results = []
    
    def calculate_performance(self, B, Q, K, use_cache=True, elapsed=0.0):
        """
        计算指定配置下的性能指标
        
        参数:
        B: batch size
        L: 层数
        Q: 查询序列长度
        K: 键值序列长度
        use_cache: 是否使用缓存
        elapsed: 推理时间(ms)
        
        返回: (总FLOPs, 总访存字节, 操作强度)
        """
        # 1. 计算总FLOPs
        flops = self.calculate_flops(B, Q, K, use_cache)
        
        # 2. 计算总访存量
        memory_access = self.calculate_memory_access(B, Q, K, use_cache)
        
        # 3. 计算操作强度
        ai = flops / memory_access if memory_access > 0 else 0
        
        # 4. 计算推理时间
        elapsed = elapsed / 1000 # 转换为秒
        
        # 存储结果
        result = {
            'B': B, 'Q': Q, 'K': K,
            'flops': flops,
            'memory_access': memory_access,
            'ai': ai,
            'elapsed': elapsed
        }
        self.results.append(result)
        
        return flops, memory_access, ai
    
    def calculate_flops(self, B, Q, K, use_cache):
        """计算总FLOPs"""
        total_flops = 0
        
        # 嵌入层FLOPs (索引查找，无浮点计算)
        
        # 每层Transformer的计算
        for _ in range(self.n_layers):
            # 多头注意力
            # QKV投影
            flops_qkv = 3 * 2 * B * Q * self.d_model * self.d_model
            # QK^T计算,多头
            flops_qk = 2 * B * self.n_heads * Q * K * self.d_head
            # softmax
            # flops_softmax = 2 * B * self.n_heads * Q * K * self.d_head``
            # 注意力加权和
            flops_av = 2 * B * self.n_heads * Q * K * self.d_head
            # 输出投影
            flops_out = 2 * B * Q * self.d_model * self.d_model
            # 残差连接
            flops_res = B * Q * self.d_model
            
            # FFN层
            flops_fc1 = 2 * B * Q * self.d_model * self.d_ff
            # GELU
            # flops_gelu = 3 * B * Q * self.d_ff
            flops_fc2 = 2 * B * Q * self.d_ff * self.d_model
            flops_res2 = B * Q * self.d_model
            
            # LayerNorm (2次)
            # flops_ln = 4 * 2 * B * Q * self.d_model
            
            # 累加本层FLOPs
            layer_flops = (
                flops_qkv + flops_qk + flops_av + flops_out + flops_res +
                flops_fc1  + flops_fc2 + flops_res2 
            )
            total_flops += layer_flops
        
        # 语言模型头
        flops_lm = 2 * B * Q * self.d_model * self.vocab_size
        total_flops += flops_lm
        
        return total_flops
    
    def calculate_memory_access(self, B, Q, K, use_cache):
        """计算总访存量 (字节)"""
        # 1. 权重访问
        # 嵌入层权重
        mem_embed = self.vocab_size * self.d_model * self.bytes_per_value
        
        # 每层权重
        # QKV投影 + 输出投影
        mem_attn_per_layer = (4 * self.d_model * self.d_model) * self.bytes_per_value
        # FFN层
        mem_ffn_per_layer = (self.d_model * self.d_ff + self.d_ff * self.d_model) * self.bytes_per_value
        
        # 语言模型头权重
        mem_lm = self.d_model * self.vocab_size * self.bytes_per_value

        # 总权重访存
        mem_weights = mem_embed + self.n_layers * (mem_attn_per_layer + mem_ffn_per_layer) + mem_lm
        
        # 2. 中间激活值，注意力分数访问

        # 嵌入层输入(读取)
        mem_embed_in = B * Q * self.vocab_size * self.bytes_per_value
        # 嵌入层输出(写入)
        mem_embed_out = B * Q * self.d_model * self.bytes_per_value
        mem_embed = mem_embed_in + mem_embed_out

        # 输入激活值(读取)
        mem_input = B * Q * self.d_model * self.bytes_per_value
        # QKV投影
        if not use_cache:
            # QKV(写入)
            mem_qkv = 3 * B * Q * self.d_model * self.d_model * self.bytes_per_value
        else:
            # Q(写入)
            mem_q =  B * Q * self.d_model * self.d_model * self.bytes_per_value
            # 键值缓存(读取)
            mem_kv_cache = 2 * B * K * self.d_model * self.bytes_per_value
            mem_qkv = mem_q + mem_kv_cache

        # QK^T计算,多头(写入)
        mem_qk = B * self.n_heads * Q * K * self.d_head * self.bytes_per_value
        # softmax,多头(写入)
        mem_softmax = B * self.n_heads * Q * K * self.d_head * self.bytes_per_value
        # 注意力加权和(写入)
        mem_av = B * self.n_heads * Q * self.d_head * self.bytes_per_value
        # 输出投影(写入)
        mem_attn_out = B * Q * self.d_model * self.bytes_per_value
        # 残差连接(写入)
        mem_res1 = B * Q * self.d_model * self.bytes_per_value
        # ffn1激活值(写入)
        mem_ffn1 = B * Q * self.d_ff * self.bytes_per_value
        # GELU(写入)
        mem_gelu = B * Q * self.d_ff * self.bytes_per_value
        # ffn2激活值(写入)
        mem_ffn2 = B * Q * self.d_model * self.bytes_per_value
        # 残差连接(写入)
        mem_res2 = B * Q * self.d_model * self.bytes_per_value
        # 单层访存
        mem_block = mem_input + mem_qkv + mem_qk + mem_softmax + mem_av + mem_attn_out + mem_res1 + mem_ffn1 + mem_gelu + mem_ffn2 + mem_res2
        # 层数
        mem_activation = self.n_layers * mem_block

        # lm头输入(读取)
        mem_lm_in = B * Q * self.d_model * self.bytes_per_value
        # lm头输出(写入)
        mem_lm_out = B * Q * self.vocab_size * self.bytes_per_value
        mem_lm = mem_lm_in + mem_lm_out

        total_memory_access = mem_embed + mem_weights + mem_activation + mem_lm
        return total_memory_access
    
    def plot_roofline(self, peak_flops=1.07e12, mem_bw=62.1e9):
        """
        绘制Roofline分析图
        
        参数:
        peak_flops: 峰值计算性能 (TFLOPS/s)
        mem_bw: 内存带宽 (GB/s)
        """
        if not self.results:
            print("No results to plot. Run calculations first.")
            return
        
        # 创建绘图
        plt.figure(figsize=(12, 8))
        plt.xscale('log')
        plt.yscale('log')
        
        # 计算Roofline曲线
        ai_range = np.logspace(-1, 3, 500)
        roofline = np.minimum(peak_flops, mem_bw * ai_range)
        
        # 绘制Roofline
        plt.plot(ai_range, roofline, 'b-', linewidth=2.5, label='Roofline Bound')
        
        # 标记拐点
        knee_point = peak_flops / mem_bw
        plt.axvline(x=knee_point, color='gray', linestyle='--', alpha=0.7)
        plt.axhline(y=peak_flops, color='gray', linestyle='--', alpha=0.7)
        
        # 绘制不同配置的性能点
        markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'D']
        colors = plt.cm.tab10.colors
        
        for i, res in enumerate(self.results):
            # 生成配置标签
            config_label = f"B={res['B']}, Q={res['Q']}"
            if res['K'] != res['Q']:
                config_label += f", K={res['K']}"
            
            # 计算理论性能 (位于Roofline上)
            theoretical_perf = min(peak_flops, res['ai'] * mem_bw)

            # 计算实际性能 (单位：FLOPS/s)
            actual_perf = res['flops'] / res['elapsed']
            
            # 绘图
            marker = markers[i % len(markers)]
            color = colors[i % len(colors)]
            plt.scatter(res['ai'], theoretical_perf, s=120, 
                        color=color, marker=marker, label=config_label)
            if actual_perf is not None:
                plt.scatter(res['ai'], actual_perf, s=120, 
                            color='red', marker='x', label=f"{config_label} (actual)")
        
        # 添加标注
        plt.text(0.15, peak_flops * 1.2, 
                 f'Comp. Peak: {peak_flops/1e12:.2f} TOps/s',
                 fontsize=12, ha='left')
        plt.text(knee_point * 15, peak_flops * 15,
                 f'Mem BW: {mem_bw/1e9:.1f} GB/s',
                 fontsize=12, ha='left')
        plt.text(0.2, peak_flops * 0.1, 
                 'Memory-Bound Region',
                 fontsize=14, rotation=40, ha='left', 
                 bbox=dict(boxstyle="round,pad=0.3", 
                          facecolor="lightyellow", 
                          edgecolor="orange", 
                          alpha=0.7))
        plt.text(15, peak_flops * 0.5, 
                 'Compute-Bound Region',
                 fontsize=14, ha='left', 
                 bbox=dict(boxstyle="round,pad=0.3", 
                          facecolor="lightblue", 
                          edgecolor="blue", 
                          alpha=0.7))
        
        # 坐标轴设置
        plt.xlabel('Operational Intensity (FLOP/Byte)', fontsize=14)
        plt.ylabel('Theoretical Performance (FLOP/s)', fontsize=14)
        plt.title('GPT-2 Roofline Performance Analysis', fontsize=16)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(fontsize=10, loc='upper left')
        
        plt.tight_layout()
        plt.savefig("roofline.png", dpi=200)
        # plt.show()
    
    def print_results(self):
        """打印所有计算结果"""
        print(f"\nGPT-2 Performance Analysis Results, bytes_per_value={self.bytes_per_value} Byte")
        print("=" * 80)
        print(f"{'Config':<25} {'FLOPs (M)':>12} {'Mem (GB)':>12} {'AI (FLOP/Byte)':>15}")
        print("-" * 80)
        
        for res in self.results:
            config = f"B={res['B']},Q={res['Q']},K={res['K']}"
            flops_g = res['flops'] / 1e6
            mem_gb = res['memory_access'] / 1e9
            print(f"{config:<25} {flops_g:12.1f} {mem_gb:12.3f} {res['ai']:15.1f}")
        
        print("=" * 80)
        print("Note: AI = Operational Intensity (FLOP/Byte)")


# ====================== 示例使用 ======================
if __name__ == "__main__":
    # 1.H200 相关数据 
    peak_flops = 1671e12 # TFLOPS FP16
    # peak_flops = 60.0e12  # TFLOPS FP32
    mem_bw = 4.8e12 # TB/s

    # # 1. 创建分析器 (使用GPT-2 默认参数)
    # analyzer = GPT2PerformanceAnalyzer(
    #     d_model=768, 
    #     n_heads=12,
    #     d_ff=3072,
    #     n_layers=12,
    #     vocab_size=50257
    # )
    
    # 2. 创建分析器 (使用GPT-2 Medium默认参数)
    analyzer = GPT2PerformanceAnalyzer(
        d_model=1024, 
        n_heads=16,
        d_ff=4096,
        n_layers=1,
        vocab_size=50257
    )
    # 2. 计算不同配置下的性能
    # 全量推理 (完整序列处理)
    analyzer.calculate_performance(B=1, Q=512, K=512, use_cache=False, elapsed=0.19388)
    analyzer.calculate_performance(B=1, Q=1024, K=1024, use_cache=False, elapsed=0.38776)
    
    # 自回归生成 (仅处理当前token)
    analyzer.calculate_performance(B=1, Q=1, K=512, use_cache=True,elapsed=0.017992)
    analyzer.calculate_performance(B=1, Q=1, K=1024, use_cache=True,elapsed=0.035984)
    

    # 3. 打印结果
    analyzer.print_results()
    
    # 4. 绘制Roofline分析图 (使用图片提供的硬件参数)
    analyzer.plot_roofline(peak_flops=peak_flops, mem_bw=mem_bw)