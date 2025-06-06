from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time
from deepspeed.profiling.flops_profiler import get_model_profile
from transformers.trainer_utils import get_last_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
last_checkpoint = get_last_checkpoint("./tmp/test-clm-xwwx-sxv")
# 初始化模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained(last_checkpoint)
model = GPT2LMHeadModel.from_pretrained(last_checkpoint, use_cache=True).to(device)

# 配置参数
# model_name = 'gpt2'  # 可替换为 'gpt2-medium' 或 'gpt2-large'
seq_lengths = [512, 1024]
batch_size = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型和分词器
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
model.eval()

# 1. 全量推理统计
print("=== 全量推理（use_cache=False） ===")
# 构造输入
input_ids = torch.randint(0, tokenizer.vocab_size, (1, 512), dtype=torch.long).to(device)
args = (input_ids,)
kwargs = {'use_cache': False}

flops, macs, params = get_model_profile(
    model,
    args=args,
    kwargs=kwargs,
    print_profile=True,
    detailed=True,
    module_depth=-1,
    top_modules=1,
    warm_up=10,
    as_string=True
)

print(f"FLOPs: {flops}")
print(f"MACs: {macs}")
print(f"Params: {params}")

# 2. 自回归推理统计
print("\n=== 自回归推理（use_cache=True） ===")

with torch.no_grad():
    past = model(input_ids, use_cache=True).past_key_values

# 2.2 单步推理
next_token = torch.randint(0, tokenizer.vocab_size, (1, 1)).to(device)

# # 构造输入
# next_token = torch.randint(0, tokenizer.vocab_size, (1, 1), dtype=torch.long).to(device)
# past_key_values = tuple([(torch.randn(1, 12, 512, 64).to(device), 
#                          torch.randn(1, 12, 512, 64).to(device)) 
#                         for _ in range(12)])

args = (next_token,)
kwargs = {
    'use_cache': True,
    'past_key_values': past
}

flops, macs, params = get_model_profile(
    model,
    args=args,
    kwargs=kwargs,
    print_profile=True,
    detailed=True,
    module_depth=-1,
    top_modules=1,
    warm_up=10,
    as_string=True
)

print(f"FLOPs: {flops}")
print(f"MACs: {macs}")
print(f"Params: {params}")

# 创建虚拟输入
def create_dummy_input(seq_len):
    return torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len)).to(device)

# 更精确的内存访问估算
# def estimate_mem_access(model, input_shape, verbose=False):
#     """改进的内存访问量估算函数"""
#     batch, seq_len = input_shape
#     hidden_size = model.config.hidden_size
#     layers = model.config.n_layer
#     vocab_size = tokenizer.vocab_size
    
#     # 参数内存访问 (加载权重)
#     param_bytes = 0
#     for name, param in model.named_parameters():
#         # 获取数据类型大小
#         dtype_size = param.element_size()
#         param_bytes += param.numel() * dtype_size
    
#     # 激活值内存访问 (分项计算)
#     # 1. 输入嵌入
#     input_emb = batch * seq_len * hidden_size * 2  # float16, 4 bytes? 但实际上用2字节模拟
#     # 2. 注意力层
#     #    - Query, Key, Value 投影: 3 * batch * seq_len * hidden_size^2
#     #    - 注意力分数: batch * layers * seq_len^2
#     attn_proj = layers * 3 * batch * seq_len * hidden_size * 2
#     attn_score = layers * batch * (seq_len ** 2) * 2
#     # 3. MLP层
#     mlp = layers * batch * seq_len * hidden_size * model.config.n_inner * 2
#     # 4. 输出层
#     output_proj = batch * seq_len * hidden_size * vocab_size * 2
    
#     # 总激活值大小
#     activation_bytes = input_emb + attn_proj + attn_score + mlp + output_proj
    
#     # 总内存访问 (读+写)
#     total_bytes = param_bytes * 2 + activation_bytes * 2
    
#     if verbose:
#         print("\n内存访问量分解:")
#         print(f"参数访问: {param_bytes/1e9:.2f} GB")
#         print(f"输入嵌入: {input_emb/1e9:.2f} GB")
#         print(f"注意力投影: {attn_proj/1e9:.2f} GB")
#         print(f"注意力分数: {attn_score/1e9:.2f} GB")
#         print(f"MLP层: {mlp/1e9:.2f} GB")
#         print(f"输出投影: {output_proj/1e9:.2f} GB")
    
#     return total_bytes

# # 测试不同序列长度
# # for seq_len in seq_lengths:
#     print(f"\n{'='*60}")
#     print(f"使用 get_model_profile 分析 - 序列长度: {seq_len}, 批量大小: {batch_size}")
    
#     # 准备虚拟输入
#     input_ids = create_dummy_input(seq_len)
    
#     # 预热GPU缓存
#     with torch.no_grad():
#         for _ in range(3):
#             model(input_ids)
    
#     # 使用get_model_profile进行分析
#     start_time = time.time()
#     with torch.no_grad():
#         flops, macs, params, details = get_model_profile(
#             model=model,
#             input_shape=(1,512),
#             print_profile=False,
#             detailed=False,
#             module_depth=1,
#             warm_up=1,
#             as_string=False,
#             output_file=None
#         )
#     elapsed = time.time() - start_time
    
#     # 估算内存访问量
#     mem_access = estimate_mem_access(model, input_ids.shape, verbose=True)
    
#     # 计算理论计算强度
#     operational_intensity = flops / mem_access  # FLOPs/byte
    
#     # 计算实际性能
#     flops_seconds = flops / elapsed  # 实际FLOPS/s
#     flops_seconds = flops_seconds / 1e9  # 转换为 GFLOPs/s
    
#     # 打印结果
#     print(f"\n{'='*60}")
#     print(f"序列长度: {seq_len}, 批量大小: {batch_size}")
#     print(f"分析耗时: {elapsed:.4f}秒")
#     print(f"FLOPs: {flops/1e9:.2f} GFLOPs")
#     print(f"MACs: {macs/1e9:.2f} GMACs")
#     print(f"内存访问量: {mem_access/1e9:.2f} GB")
#     print(f"计算强度: {operational_intensity:.2f} FLOPs/byte")
#     print(f"参数总量: {params/1e6:.2f} Million")
#     print(f"实测性能: {flops_seconds:.2f} GFLOPs/s")
    
#     # 输出详细模型分析
#     print("\n详细模型分析:")
#     for line in details:
#         print(line)
    
#     # 显存清理
#     del input_ids
#     torch.cuda.empty_cache()