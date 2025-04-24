from transformers import AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint
import torch

# 1. 加载预训练模型结构（如果检查点不包含模型配置）
model = AutoModelForCausalLM.from_pretrained("../../gpt2")

# 2. 获取最新的检查点路径
# last_checkpoint = get_last_checkpoint("./tmp/test-clm-ori")
last_checkpoint = get_last_checkpoint("./tmp/test-clm_")
print(f"找到最新检查点：{last_checkpoint}")

# 3. 加载检查点参数到模型
if last_checkpoint is not None:
    # 方法1：直接加载整个检查点目录（自动包含参数和配置）
    model = AutoModelForCausalLM.from_pretrained(last_checkpoint)
    
    # 方法2：仅加载参数（需手动加载.bin文件）
    # checkpoint = torch.load(f"{last_checkpoint}/pytorch_model.bin", map_location='cpu')
    # model.load_state_dict(checkpoint, strict=False)
else:
    raise ValueError("未找到检查点")

# # 4. 查看参数示例（打印第一层的权重）
# for name, param in model.named_parameters():
#     if 'weight' in name:  # 过滤出权重参数
#         print(f"参数层：{name}")

#         print(f"形状：{param.shape}")
#         print(f"前5个值：{param.data[0][:5]}\n")  # 打印第一行前5个值
#         break  # 只显示一个示例

# 访问第0层注意力中的 Q/K/V 投影权重（形状：768, 2304）
layer0_attn = model.transformer.h[0].attn.c_attn
print(layer0_attn)
print("Q/K/V投影权重：", layer0_attn.weight.shape)  # 输出: torch.Size([768, 2304])

# 访问第0层注意力输出的投影权重（形状：768, 768）
layer0_attn_proj = model.transformer.h[0].attn.c_proj
print(layer0_attn_proj)
print("输出投影权重：", layer0_attn_proj.weight.shape)  # 输出: torch.Size([768, 768])

# 查看具体数值（例如第一行前5个值）
print(layer0_attn.weight.data[:, 10:])  # 输出张量值