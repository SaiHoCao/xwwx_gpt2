#!/usr/bin/env python3

import os
import re
import subprocess
import time
import sys

def update_threshold(percentage):
    """
    更新 run_qa.sh 中的输出目录和 model_gpt2.py 中的阈值百分比
    
    参数:
        percentage: 整数百分比值 (例如 75, 76, 等)
    """
    percentage_float = percentage / 100.0
    
    # 1. 更新 run_qa.sh
    with open('run_qa.sh', 'r') as f:
        content = f.read()
    
    # 使用正则表达式更新输出目录路径
    updated_content = re.sub(
        r'--output_dir\s+\./tmp/0_medium_squad_ori_\d+_eval',
        f'--output_dir ./tmp/0_medium_squad_ori_{percentage}_eval',
        content
    )
    
    with open('run_qa.sh', 'w') as f:
        f.write(updated_content)
    
    print(f"✅ 已更新 run_qa.sh 的输出目录为 ./tmp/0_medium_squad_ori_{percentage}_eval")
    
    # 2. 更新 model_gpt2.py
    model_file = 'model_gpt2.py'
    if not os.path.exists(model_file):
        print(f"⚠️ 警告: 找不到 {model_file}，只更新了 run_qa.sh")
        return
    
    with open(model_file, 'r') as f:
        model_content = f.read()
    
    # 使用正则表达式更新阈值百分比
    updated_model_content = re.sub(
        r'threshold\s*=\s*torch\.quantile\(inputx_kv\.abs\(\)\.flatten\(\),\s*0\.\d+\)',
        f'threshold = torch.quantile(inputx_kv.abs().flatten(), {percentage_float:.2f})',
        model_content
    )
    
    with open(model_file, 'w') as f:
        f.write(updated_model_content)
    
    print(f"✅ 已更新 model_gpt2.py 中的阈值百分比为 {percentage_float:.2f}")

def run_evaluation():
    """
    运行 run_qa.sh 进行评估
    """
    print("🚀 开始运行评估...")
    result = subprocess.run(['bash', 'run_qa.sh'], capture_output=True, text=True)
    
    # 提取评估结果
    output = result.stdout + result.stderr
    
    # 寻找评估指标
    exact_match = re.search(r'eval_exact_match\s*=\s*(\d+\.\d+)', output)
    f1_score = re.search(r'eval_f1\s*=\s*(\d+\.\d+)', output)
    
    if exact_match and f1_score:
        em = exact_match.group(1)
        f1 = f1_score.group(1)
        print(f"📊 评估结果: Exact Match = {em}, F1 Score = {f1}")
    else:
        print("⚠️ 无法从输出中提取评估指标")
    
    print("✅ 评估完成")
    
    return output

def main():
    if len(sys.argv) < 3:
        print("用法: python run_sequential_eval.py <开始百分比> <结束百分比>")
        print("例如: python run_sequential_eval.py 75 80")
        sys.exit(1)
    
    try:
        start_percentage = int(sys.argv[1])
        end_percentage = int(sys.argv[2])
        
        if start_percentage < 0 or start_percentage > 100 or end_percentage < 0 or end_percentage > 100:
            print("错误: 百分比必须在0到100之间")
            sys.exit(1)
            
        if start_percentage > end_percentage:
            print("错误: 开始百分比必须小于或等于结束百分比")
            sys.exit(1)
    except ValueError:
        print("错误: 请提供有效的整数百分比")
        sys.exit(1)
    
    # 创建结果日志文件
    log_file = f"eval_results_{start_percentage}_{end_percentage}.log"
    with open(log_file, 'w') as f:
        f.write(f"阈值评估结果 ({start_percentage}% - {end_percentage}%)\n")
        f.write("=" * 50 + "\n\n")
    
    # 依次运行每个百分比的评估
    for percentage in range(start_percentage, end_percentage + 1):
        print("\n" + "=" * 60)
        print(f"🔄 开始运行阈值为 {percentage}% 的评估 ({percentage-start_percentage+1}/{end_percentage-start_percentage+1})")
        print("=" * 60 + "\n")
        
        # 更新阈值和输出目录
        update_threshold(percentage)
        
        # 运行评估
        start_time = time.time()
        output = run_evaluation()
        end_time = time.time()
        
        # 提取评估指标
        exact_match = re.search(r'eval_exact_match\s*=\s*(\d+\.\d+)', output)
        f1_score = re.search(r'eval_f1\s*=\s*(\d+\.\d+)', output)
        
        # 记录结果到日志
        with open(log_file, 'a') as f:
            f.write(f"阈值: {percentage}%\n")
            f.write(f"运行时间: {end_time - start_time:.2f} 秒\n")
            
            if exact_match and f1_score:
                em = exact_match.group(1)
                f1 = f1_score.group(1)
                f.write(f"Exact Match: {em}\n")
                f.write(f"F1 Score: {f1}\n")
            else:
                f.write("无法提取评估指标\n")
                
            f.write("\n" + "-" * 30 + "\n\n")
        
        print(f"⏱️ 耗时: {end_time - start_time:.2f} 秒")
        print(f"📝 结果已保存到 {log_file}")
    
    print("\n" + "=" * 60)
    print(f"🎉 所有评估已完成! 总共运行了 {end_percentage-start_percentage+1} 个阈值的评估")
    print(f"📊 详细结果已保存到 {log_file}")

if __name__ == "__main__":
    main()