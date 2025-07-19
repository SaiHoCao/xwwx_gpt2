import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt

# # ====== 你只需修改下面这几个变量 ======
# BASE_DIR = "./tmp/medium-xwwxsxv-copa/"           # 基础目录
# PATTERN_PREFIX = "medium_xwwxsxv_copa"            # 文件夹前缀
# METRIC_KEYS = ["eval_accuracy"]                   # 要收集的指标名列表
# OUTPUT_DIR = "./results"                          # 结果保存目录
# OUTPUT_PREFIX = "medium_xwwxsxv_copa"             # 输出文件前缀
# PLOT_TITLE = None                                  # 图表标题（可选）
# # ====================================

# # ====== 你只需修改下面这几个变量 ======
# BASE_DIR = "./tmp/medium-ori-copa/"           # 基础目录
# PATTERN_PREFIX = "medium_ori_copa"            # 文件夹前缀
# METRIC_KEYS = ["eval_accuracy"]                   # 要收集的指标名列表
# OUTPUT_DIR = "./results"                          # 结果保存目录
# OUTPUT_PREFIX = "medium_ori_copa"             # 输出文件前缀
# PLOT_TITLE = None                                  # 图表标题（可选）
# # ====================================

# # ====== 你只需修改下面这几个变量 ======
# BASE_DIR = "./tmp/medium-xwwxsxv-piqa/"           # 基础目录
# PATTERN_PREFIX = "medium_xwwxsxv_piqa"            # 文件夹前缀
# METRIC_KEYS = ["eval_accuracy"]                   # 要收集的指标名列表
# OUTPUT_DIR = "./results"                          # 结果保存目录
# OUTPUT_PREFIX = "medium_xwwxsxv_piqa"             # 输出文件前缀
# PLOT_TITLE = None                                  # 图表标题（可选）
# # ====================================

# # ====== 你只需修改下面这几个变量 ======
# BASE_DIR = "./tmp/medium-ori-piqa/"           # 基础目录
# PATTERN_PREFIX = "medium_ori_piqa"            # 文件夹前缀
# METRIC_KEYS = ["eval_accuracy"]                   # 要收集的指标名列表
# OUTPUT_DIR = "./results"                          # 结果保存目录
# OUTPUT_PREFIX = "medium_ori_piqa"             # 输出文件前缀
# PLOT_TITLE = None                                  # 图表标题（可选）
# # ====================================

# # ====== 你只需修改下面这几个变量 ======
# BASE_DIR = "./tmp/medium-xwwxsxv-wiki2/"           # 基础目录
# PATTERN_PREFIX = "medium_xwwxsxv_wiki2"            # 文件夹前缀
# METRIC_KEYS = ["perplexity"]                       # 要收集的指标名列表
# OUTPUT_DIR = "./results"                          # 结果保存目录
# OUTPUT_PREFIX = "medium_xwwxsxv_wiki2"             # 输出文件前缀
# PLOT_TITLE = None                                  # 图表标题（可选）
# # ====================================

# # ====== 你只需修改下面这几个变量 ======
# BASE_DIR = "./tmp/medium-ori-wiki2/"           # 基础目录
# PATTERN_PREFIX = "medium_ori_wiki2"            # 文件夹前缀
# METRIC_KEYS = ["perplexity"]                       # 要收集的指标名列表
# OUTPUT_DIR = "./results"                          # 结果保存目录
# OUTPUT_PREFIX = "medium_ori_wiki2"             # 输出文件前缀
# PLOT_TITLE = None                                  # 图表标题（可选）
# # ====================================

# ====== 你只需修改下面这几个变量 ======
BASE_DIR = "./tmp/medium-ori-squad/"           # 基础目录
PATTERN_PREFIX = "medium_ori_squad"            # 文件夹前缀
METRIC_KEYS = ["eval_exact_match","eval_f1"]                       # 要收集的指标名列表
OUTPUT_DIR = "./results"                          # 结果保存目录
OUTPUT_PREFIX = "medium_ori_squad"             # 输出文件前缀
PLOT_TITLE = None                                  # 图表标题（可选）
# ====================================

# ====== 你只需修改下面这几个变量 ======
BASE_DIR = "./tmp/medium-xwwxsxv-squad/"           # 基础目录
PATTERN_PREFIX = "medium_xwwxsxv_squad"            # 文件夹前缀
METRIC_KEYS = ["eval_exact_match","eval_f1"]                       # 要收集的指标名列表
OUTPUT_DIR = "./results"                          # 结果保存目录
OUTPUT_PREFIX = "medium_xwwxsxv_squad"             # 输出文件前缀
PLOT_TITLE = None                                  # 图表标题（可选）
# ====================================

def collect_metrics(base_dir, pattern_prefix, metric_keys):
    pattern = re.compile(f"{re.escape(pattern_prefix)}_(\\d+)_eval")
    results = []
    for dirname in os.listdir(base_dir):
        match = pattern.match(dirname)
        if match:
            sparsity = int(match.group(1))
            result_file = os.path.join(base_dir, dirname, "eval_results.json")
            if os.path.exists(result_file):
                try:
                    with open(result_file, "r") as f:
                        data = json.load(f)
                    
                    # 收集所有指定的指标
                    result_row = {"sparsity": sparsity}
                    for metric_key in metric_keys:
                        value = data.get(metric_key, None)
                        result_row[metric_key] = value
                    
                    results.append(result_row)

                except Exception as e:
                    print(f"⚠️ 读取 {result_file} 时出错: {str(e)}")
    results.sort(key=lambda x: x["sparsity"])
    return results

def save_to_csv(results, metric_keys, output_dir, output_prefix):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    
    # 保存包含所有指标的CSV文件
    output_file = os.path.join(output_dir, f"{output_prefix}_sparsity_metrics.csv")
    df.to_csv(output_file, index=False)
    print(f"结果已保存到 {output_file}")
    
    # 只有当只有一个指标时，才保存单独的CSV文件
    # if len(metric_keys) == 1:
    #     metric_key = metric_keys[0]
    #     if metric_key in df.columns:
    #         metric_df = df[["sparsity", metric_key]].dropna()
    #         metric_output_file = os.path.join(output_dir, f"{output_prefix}_sparsity_{metric_key}.csv")
    #         metric_df.to_csv(metric_output_file, index=False)
    #         print(f"指标 {metric_key} 结果已保存到 {metric_output_file}")
    
    return df

def plot_metrics(df, metric_keys, output_dir, output_prefix, title=None):
    if df is None or df.empty:
        print("没有数据可以绘图")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果只有一个指标，创建单独的图表
    if len(metric_keys) == 1:
        metric_key = metric_keys[0]
        if metric_key not in df.columns:
            print(f"⚠️ 指标 {metric_key} 不在数据中，跳过绘图")
            return
            
        # 过滤掉空值
        plot_df = df[["sparsity", metric_key]].dropna()
        if plot_df.empty:
            print(f"⚠️ 指标 {metric_key} 没有有效数据，跳过绘图")
            return
            
        output_file = os.path.join(output_dir, f"{output_prefix}_sparsity_{metric_key}.png")
        plt.figure(figsize=(10, 6))
        plt.plot(plot_df["sparsity"], plot_df[metric_key], marker='o', label=metric_key)
        plt.xlabel("Sparsity (%)")
        plt.ylabel(metric_key)
        plt.legend()
        if title is None:
            title = f"{output_prefix} Sparsity vs {metric_key}"
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"图表已保存到 {output_file}")
        plt.show()
    
    # 如果有多个指标，只创建组合图表
    else:
        # 检查哪些指标有数据
        available_metrics = [key for key in metric_keys if key in df.columns and not df[key].isna().all()]
        
        if len(available_metrics) > 1:
            output_file = os.path.join(output_dir, f"{output_prefix}_sparsity_metrics.png")
            plt.figure(figsize=(12, 8))
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            for i, metric_key in enumerate(available_metrics):
                plot_df = df[["sparsity", metric_key]].dropna()
                if not plot_df.empty:
                    color = colors[i % len(colors)]
                    plt.plot(plot_df["sparsity"], plot_df[metric_key], marker='o', label=metric_key, 
                            linewidth=2, markersize=6, color=color)
            
            plt.xlabel("Sparsity (%)")
            plt.ylabel("Metric Values")
            plt.legend()
            if title is None:
                title = f"{output_prefix} Sparsity vs Metrics"
            plt.title(title)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(output_file)
            print(f"组合图表已保存到 {output_file}")
            plt.show()
        elif len(available_metrics) == 1:
            print(f"⚠️ 只有一个指标 {available_metrics[0]} 有数据，跳过组合图表")
        else:
            print("⚠️ 没有找到任何有效数据的指标")

if __name__ == "__main__":
    results = collect_metrics(BASE_DIR, PATTERN_PREFIX, METRIC_KEYS)
    df = save_to_csv(results, METRIC_KEYS, OUTPUT_DIR, OUTPUT_PREFIX)
    plot_metrics(df, METRIC_KEYS, OUTPUT_DIR, OUTPUT_PREFIX, PLOT_TITLE) 