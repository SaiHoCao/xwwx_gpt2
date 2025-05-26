import matplotlib.pyplot as plt
import numpy as np

# sparsities = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
# xwwx_ppl = [21.1049, 21.1487, 21.4105, 22.0654, 23.2605, 25.2468, 28.237, 32.5833, 38.7605, 47.277, 57.4027]
# ori_ppl  = [21.1049, 21.1497, 21.4059, 22.0564, 23.2589, 25.2452, 28.259, 32.5606, 38.8177, 47.2808, 57.3932]

sparsities = [0, 10, 20, 30, 40, 50, 60, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98]
xwwx_ppl = [21.1049, 21.1124, 21.1396, 21.2209, 21.3887, 21.7158, 22.3004, 23.4604, 23.6035, 23.7968, 24.003, 24.2277, 24.4489, 24.711, 25.0103, 25.3251, 25.6764, 26.086, 26.5018, 27.0274, 27.6074, 28.3403, 29.1539, 30.0279, 31.0182, 32.2939, 33.7805, 35.5199, 37.8724, 40.5717, 44.1183, 48.7415, 54.845, 63.3303, 76.1484, 97.734]
ori_ppl  = [21.1049, 21.1115, 21.1402, 21.2124, 21.3866, 21.7158, 22.2983, 23.4494, 23.6184, 23.7997, 24.0031, 24.2093, 24.4309, 24.7013, 25.0173, 25.3195, 25.675, 26.0573, 26.5121, 27.0414, 27.6197, 28.318, 29.1315, 30.0422, 31.0796, 32.2801, 33.7333, 35.5406, 37.8273, 40.6079, 44.1082, 48.7387, 54.7316, 63.2537, 76.1988, 97.8877]

# 分段拉伸函数
def map_sparsity(x):
    if x <= 60:
        return x * 0.5  # 0-70区间压缩
    else:
        return 30 + (x - 60) * 2  # 70-100区间拉伸

mapped_x = [map_sparsity(x) for x in sparsities]

# 创建图形
plt.figure(figsize=(12, 6))

# 绘制折线
plt.plot(mapped_x, xwwx_ppl, 'b-o', label='XWWX Model', linewidth=2, markersize=7, alpha=0.9, linestyle='-')
plt.plot(mapped_x, ori_ppl, 'r--s', label='ORI Model', linewidth=2, markersize=6, alpha=0.9, linestyle='--')

# for x, y in zip(sparsities, xwwx_ppl):
#     plt.text(x, y, f'{y:.4f}', color='blue', fontsize=9, ha='left', va='bottom')
# for x, y in zip(sparsities, ori_ppl):
#     plt.text(x, y, f'{y:.4f}', color='red', fontsize=9, ha='right', va='top')

# 设置标题和标签
plt.title('Perplexity vs Sparsity', fontsize=14)
plt.xlabel('Sparsity (%)', fontsize=12)
plt.ylabel('Perplexity', fontsize=12)

# 添加网格
plt.grid(True, linestyle='--', alpha=0.7)

# 添加图例
plt.legend(fontsize=10)

# 设置x轴刻度和标签
xticks = [0, 10, 20, 30, 40, 50, 60, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98]
xtick_labels = [str(x) for x in xticks]
plt.xticks([map_sparsity(x) for x in xticks], xtick_labels)

# 保存图片
plt.savefig('perplexity_comparison_eval.png', dpi=300, bbox_inches='tight')
plt.close() 