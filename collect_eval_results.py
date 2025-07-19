#!/usr/bin/env python3

import os
import json
import re
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import argparse

def collect_results(pattern_str, base_dir="./tmp"):
    """
    收集指定目录下符合模式的评估结果
    提取稀疏度和评估指标
    
    参数:
        pattern_str: 目录名模式，形如 "0_medium_squad_ori"
        base_dir: 基础目录路径
    """
    import numpy as np
    results = []
    all_metrics = set()
    
    # 检查基础目录是否存在
    if not os.path.exists(base_dir):
        print(f"错误: {base_dir} 目录不存在")
        return results, []
    
    # 构建正则表达式模式
    # 将 pattern_str 中的特殊字符转义
    escaped_pattern = re.escape(pattern_str)
    # 构建完整的正则表达式，匹配后面的数字和"_eval"
    pattern = re.compile(f'{escaped_pattern}_(\\d+)_eval')
    
    print(f"🔍 使用模式 '{pattern_str}_xxx_eval' 在 {base_dir} 中搜索...")
    
    # 遍历指定目录下的所有文件夹
    for dirname in os.listdir(base_dir):
        match = pattern.match(dirname)
        if match:
            # 提取稀疏度
            sparsity = int(match.group(1))
            
            # 构建评估结果文件路径
            result_file = os.path.join(base_dir, dirname, "eval_results.json")
            
            # 检查结果文件是否存在
            if os.path.exists(result_file):
                try:
                    # 读取评估结果
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    # 只从指定的key中选取
                    metric_keys = ["eval_accuracy", "eval_exact_match", "eval_f1_score"]
                    metrics = {k: round(v, 3) for k, v in data.items() if k in metric_keys}
                    all_metrics.update(metrics.keys())
                    # 结果字典
                    result = {"sparsity": sparsity}
                    result.update(metrics)
                    results.append(result)
                    print(f"✅ 已读取 {dirname} 的评估结果")
                except Exception as e:
                    print(f"⚠️ 读取 {result_file} 时出错: {str(e)}")
            else:
                print(f"⚠️ 找不到评估结果文件: {result_file}")
    # 补齐缺失的指标为 np.nan
    for r in results:
        for m in all_metrics:
            if m not in r:
                r[m] = np.nan
    results.sort(key=lambda x: x["sparsity"])
    return results, list(all_metrics)

def save_results(results, output_prefix, output_dir="./results"):
    """
    将结果保存为CSV文件
    
    参数:
        results: 结果列表
        output_prefix: 输出文件前缀
        output_dir: 输出目录
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{output_prefix}_sparsity_results.csv")
    
    if not results:
        print("没有找到任何评估结果")
        return None
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 保存到CSV
    df.to_csv(output_file, index=False)
    print(f"📝 结果已保存到 {output_file}")
    
    # 打印表格形式的结果
    print("\n" + "=" * 50)
    print(f"{output_prefix} 评估结果汇总:")
    print(tabulate(df, headers="keys", tablefmt="grid", floatfmt=".3f"))
    print("=" * 50)
    
    return df

def plot_results(df, output_prefix, title=None, output_dir="./results"):
    """
    绘制稀疏度与性能关系图
    
    参数:
        df: 数据DataFrame
        output_prefix: 输出文件前缀
        title: 图表标题（可选）
        output_dir: 输出目录
    """
    if df is None or df.empty:
        print("没有数据可以绘图")
        return
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{output_prefix}_sparsity_plot.png")
    plt.figure(figsize=(10, 6))
    
    # 自动遍历所有指标（除了sparsity）
    metric_cols = [col for col in df.columns if col != "sparsity"]
    for col in metric_cols:
        plt.plot(df["sparsity"], df[col], marker='o', label=col)
    plt.xlabel("Sparsity (%)")
    plt.ylabel("Score")
    plt.legend()
    if title is None:
        title = f"{output_prefix} Sparsity vs Metrics"
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"📊 性能图表已保存到 {output_file}")
    plt.show()

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="收集和分析模型评估结果")
    
    parser.add_argument("--pattern", type=str, default="0_medium_squad_ori",
                        help="目录名模式，例如 '0_medium_squad_ori' 或 '0_medium_squad_xwwxsxv'")
    
    parser.add_argument("--base-dir", type=str, default="./tmp",
                        help="搜索评估结果的基础目录")
    
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="保存结果的目录")
    
    parser.add_argument("--output-prefix", type=str, default=None,
                        help="输出文件前缀，默认使用模式名称")
    
    parser.add_argument("--title", type=str, default=None,
                        help="图表标题，默认为 '<prefix> Sparsity vs Performance'")
    
    parser.add_argument("--no-plot", action="store_true",
                        help="不生成图表，只收集数据")
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置输出前缀
    if args.output_prefix is None:
        # 从模式中提取有意义的部分作为默认前缀
        parts = args.pattern.split("_")
        if len(parts) >= 3:
            args.output_prefix = "_".join(parts[1:])  # 去掉开头的数字
        else:
            args.output_prefix = args.pattern
    
    print(f"🚀 开始为模式 '{args.pattern}' 收集评估结果...")
    
    # 收集结果
    results, all_metrics = collect_results(args.pattern, args.base_dir)
    
    if not results:
        print(f"⚠️ 没有找到匹配模式 '{args.pattern}' 的评估结果")
        return
    
    # 保存结果
    df = save_results(results, args.output_prefix, args.output_dir)
    
    # 绘制图表
    if not args.no_plot:
        try:
            import matplotlib
            plot_results(df, args.output_prefix, args.title, args.output_dir)
        except ImportError:
            print("⚠️ 未安装matplotlib，无法生成图表。可以使用pip install matplotlib安装。")

if __name__ == "__main__":
    main()