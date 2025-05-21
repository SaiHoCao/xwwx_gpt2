import matplotlib.pyplot as plt
import numpy as np

sparsities = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
xwwx_ppl = [21.1049, 21.1487, 21.4105, 22.0654, 23.2605, 25.2468, 28.237, 32.5833, 38.7605, 47.277, 57.4027]
ori_ppl  = [21.1049, 21.1497, 21.4059, 22.0564, 23.2589, 25.2452, 28.259, 32.5606, 38.8177, 47.2808, 57.3932]

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制折线
plt.plot(sparsities, xwwx_ppl, 'b-o', label='XWWX Model', linewidth=2)
plt.plot(sparsities, ori_ppl, 'r-o', label='ORI Model', linewidth=2)

for x, y in zip(sparsities, xwwx_ppl):
    plt.text(x, y, f'{y:.4f}', color='blue', fontsize=9, ha='left', va='bottom')
for x, y in zip(sparsities, ori_ppl):
    plt.text(x, y, f'{y:.4f}', color='red', fontsize=9, ha='right', va='top')

# 设置标题和标签
plt.title('Perplexity vs Sparsity', fontsize=14)
plt.xlabel('Sparsity', fontsize=12)
plt.ylabel('Perplexity', fontsize=12)

# 添加网格
plt.grid(True, linestyle='--', alpha=0.7)

# 添加图例
plt.legend(fontsize=10)

# 设置x轴刻度
plt.xticks(sparsities)

# 保存图片
plt.savefig('perplexity_comparison_eval.png', dpi=300, bbox_inches='tight')
plt.close() 