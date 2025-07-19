import pandas as pd
import matplotlib.pyplot as plt
import os

# # ====== 你只需修改下面这几个变量 ======
# MODEL_NAMES = ["ori", "xwwxsxv"]  # 模型名列表
# TASK = "copa"                     # 任务名
# METRIC = "eval_accuracy"          # 指标名
# RESULTS_DIR = "./results"         # csv文件所在目录
# OUTPUT_PREFIX = "copa_eval_accuracy"  # 输出图片前缀
# PLOT_TITLE = "COPA accuracy in different sparsity on GPT2-Medium"      # 图表标题
# # ====================================

# # ====== 你只需修改下面这几个变量 ======
# MODEL_NAMES = ["ori", "xwwxsxv"]  # 模型名列表
# TASK = "piqa"                     # 任务名
# METRIC = "eval_accuracy"          # 指标名
# RESULTS_DIR = "./results"         # csv文件所在目录
# OUTPUT_PREFIX = "piqa_eval_accuracy"  # 输出图片前缀
# PLOT_TITLE = "PIQA accuracy in different sparsity on GPT2-Medium"      # 图表标题
# # ====================================

# # ====== 你只需修改下面这几个变量 ======
# MODEL_NAMES = ["ori", "xwwxsxv"]  # 模型名列表
# TASK = "wiki2"                     # 任务名
# METRIC = "perplexity"          # 指标名
# RESULTS_DIR = "./results"         # csv文件所在目录
# OUTPUT_PREFIX = "wiki2_perplexity"  # 输出图片前缀
# PLOT_TITLE = "WikiText2 perplexity in different sparsity on GPT2-Medium"      # 图表标题
# # ====================================

# ====== 你只需修改下面这几个变量 ======
MODEL_NAMES = ["ori", "xwwxsxv"]  # 模型名列表
TASK = "squad"                     # 任务名
METRICS = ["eval_exact_match", "eval_f1"]        # 指标名
RESULTS_DIR = "./results"         # csv文件所在目录
OUTPUT_PREFIX = "squad_metrics"  # 输出图片前缀
PLOT_TITLE = "SQuAD_metrics in different sparsity on GPT2-Medium"      # 图表标题

# ====================================

# csv_files = [
#     os.path.join(RESULTS_DIR, f"medium_{model}_{TASK}_sparsity_{METRIC}.csv")
#     for model in MODEL_NAMES
# ]
# labels = MODEL_NAMES

# plt.figure(figsize=(10, 6))
# for csv_file, label in zip(csv_files, labels):
#     if not os.path.exists(csv_file):
#         print(f"文件不存在: {csv_file}")
#         continue
#     df = pd.read_csv(csv_file)
#     plt.plot(df["sparsity"], df[METRIC], marker='o', label=label)
# 颜色和marker映射
model_colors = {
    "ori": "#1f77b4",      # 蓝色
    "xwwxsxv": "#ff7f0e",  # 橙色
    # 可以继续添加更多模型
}
metric_markers = {
    "eval_exact_match": "^",    # 三角形
    "eval_f1": "o",       # 圆形
    # 可以继续添加更多指标
}
plt.figure(figsize=(10, 6))
for model in MODEL_NAMES:
    csv_file = os.path.join(RESULTS_DIR, f"medium_{model}_{TASK}_sparsity_metrics.csv")
    if not os.path.exists(csv_file):
        print(f"文件不存在: {csv_file}")
        continue
    df = pd.read_csv(csv_file)
    color = model_colors.get(model, None)
    for metric in METRICS:
        if metric not in df.columns:
            print(f"指标 {metric} 不在 {csv_file} 中")
            continue
        marker = metric_markers.get(metric, "o")
        plt.plot(
            df["sparsity"], df[metric],
            marker=marker,
            color=color,
            label=f"{model}-{metric}",
            linewidth=2,
            markersize=7
        )

plt.xlabel("Sparsity (%)")
plt.ylabel("Metric Value")
plt.title(PLOT_TITLE)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
output_file = os.path.join(RESULTS_DIR, f"compare_{OUTPUT_PREFIX}.png")
plt.savefig(output_file)
print(f"对比图已保存到 {output_file}")
plt.show()