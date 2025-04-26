from transformers import AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint
import torch
import numpy as np
import matplotlib.pyplot as plt

def load_model_from_checkpoint(checkpoint_path):
    # 获取最新的检查点路径
    last_checkpoint = get_last_checkpoint(checkpoint_path)
    print(f"找到最新检查点：{last_checkpoint}")

    # 加载检查点参数到模型
    if last_checkpoint is not None:
        model = AutoModelForCausalLM.from_pretrained(last_checkpoint)
        return model
    else:
        raise ValueError(f"未在 {checkpoint_path} 找到检查点")

# 加载两个模型
model_ori = load_model_from_checkpoint("./tmp/test-clm-ori")
model_xwwx = load_model_from_checkpoint("./tmp/test-clm-xwwx")

# 比较模型权重
def compare_weights(model1, model2, model1_name="ori", model2_name="XWWX"):
    print(f"\n===== {model1_name} vs {model2_name} weight =====")
    
    differences = []
    param_names = []
    
    # 遍历所有参数并比较
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 != name2:
            print(f"警告：参数名不匹配: {name1} vs {name2}")
            continue
            
        # 计算差异
        diff = torch.abs(param1 - param2)
        avg_diff = torch.mean(diff).item()
        max_diff = torch.max(diff).item()
        
        # 记录有意义的差异
        if avg_diff > 1e-6:  # 忽略非常小的差异
            differences.append((name1, avg_diff, max_diff))
            param_names.append(name1)
            
    # 排序并打印差异最大的参数
    differences.sort(key=lambda x: x[1], reverse=True)
    
    if differences:
        print(f"\n差异最大的前10个参数:")
        for i, (name, avg_diff, max_diff) in enumerate(differences[:10]):
            print(f"{i+1}. {name}: 平均差异={avg_diff:.6f}, 最大差异={max_diff:.6f}")
    else:
        print("未发现显著差异")
    
    return differences, param_names

# 执行比较
differences, param_names = compare_weights(model_ori, model_xwwx)

# 可视化一些差异
def visualize_weight_difference(model1, model2, param_name, model1_name="ori", model2_name="XWWX"):
    # 获取指定参数
    param1 = dict(model1.named_parameters())[param_name].detach().cpu().numpy().flatten()
    param2 = dict(model2.named_parameters())[param_name].detach().cpu().numpy().flatten()
    
    # 计算差异
    diff = np.abs(param1 - param2)
    
    # 创建图表
    plt.figure(figsize=(15, 5))
    
    # 绘制权重分布直方图
    plt.subplot(1, 3, 1)
    plt.hist(param1, bins=50, alpha=0.5, label=model1_name)
    plt.hist(param2, bins=50, alpha=0.5, label=model2_name)
    plt.legend()
    plt.title(f"{param_name} weight")
    
    # 绘制差异直方图
    plt.subplot(1, 3, 2)
    plt.hist(diff, bins=50)
    plt.title(f"{param_name} diff")
    
    # 绘制前100个值的对比
    plt.subplot(1, 3, 3)
    sample_size = min(100, len(param1))
    indices = np.arange(sample_size)
    plt.plot(indices, param1[:sample_size], label=model1_name)
    plt.plot(indices, param2[:sample_size], label=model2_name)
    plt.legend()
    plt.title(f"{param_name} {sample_size}values diff")
    
    plt.tight_layout()
    plt.savefig(f"{param_name.replace('.', '_')}_comparison.png")
    print(f"已保存 {param_name} 的比较图表")

# 如果找到差异，可视化前三个差异最大的参数
if differences:
    for name, _, _ in differences[:3]:
        visualize_weight_difference(model_ori, model_xwwx, name)

# 还可以比较特定层的注意力权重
def compare_attention_weights(model1, model2, layer_idx=0, model1_name="ori", model2_name="XWWX"):
    # 提取第layer_idx层的注意力权重
    attn1_qkv = model1.transformer.h[layer_idx].attn.c_attn.weight
    attn2_qkv = model2.transformer.h[layer_idx].attn.c_attn.weight
    
    attn1_proj = model1.transformer.h[layer_idx].attn.c_proj.weight
    attn2_proj = model2.transformer.h[layer_idx].attn.c_proj.weight
    
    # 计算差异
    qkv_diff = torch.abs(attn1_qkv - attn2_qkv)
    proj_diff = torch.abs(attn1_proj - attn2_proj)
    
    print(f"\n===== 第{layer_idx}层注意力权重比较 =====")
    print(f"Q/K/V投影权重差异: 平均={torch.mean(qkv_diff).item():.6f}, 最大={torch.max(qkv_diff).item():.6f}")
    print(f"输出投影权重差异: 平均={torch.mean(proj_diff).item():.6f}, 最大={torch.max(proj_diff).item():.6f}")
    
    # 打印部分具体数据示例
    sample_size = 5
    print(f"\n{model1_name}模型 Q/K/V 权重示例 (前{sample_size}行，前5列):")
    print(attn1_qkv[:sample_size, :5].detach().cpu().numpy())
    
    print(f"\n{model2_name}模型 Q/K/V 权重示例 (前{sample_size}行，前5列):")
    print(attn2_qkv[:sample_size, :5].detach().cpu().numpy())
    
    print(f"\n权重差异 (前{sample_size}行，前5列):")
    print(qkv_diff[:sample_size, :5].detach().cpu().numpy())
    
    # 打印最大差异的位置和值
    max_diff_idx = torch.argmax(qkv_diff)
    max_row, max_col = max_diff_idx // qkv_diff.shape[1], max_diff_idx % qkv_diff.shape[1]
    print(f"\n最大差异位置: 行={max_row.item()}, 列={max_col.item()}")
    print(f"最大差异值: {qkv_diff[max_row, max_col].item():.6f}")
    print(f"{model1_name}值: {attn1_qkv[max_row, max_col].item():.6f}")
    print(f"{model2_name}值: {attn2_qkv[max_row, max_col].item():.6f}")

# 比较第0层的注意力权重
compare_attention_weights(model_ori, model_xwwx, layer_idx=0)

print("\n分析完成！")