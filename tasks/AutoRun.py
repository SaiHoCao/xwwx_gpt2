#!/usr/bin/env python3
import os
import re
import subprocess
import time

# 配置参数 copa
# SHELL_FILE = 'run_copa.sh'
# MODEL_FILE = '../model_gpt2.py'
# OUTPUT_DIR_TEMPLATE = '../tmp/medium-xwwxsxv-copa/medium_xwwxsxv_copa_{sparsity}_eval'
# LOG_FILE = 'copa_xwwxsxv_eval_results.log'
# EVAL_METRICS = ['eval_accuracy']

# 配置参数 wiki
# SHELL_FILE = 'run_clm.sh'
# MODEL_FILE = '../model_gpt2.py'
# OUTPUT_DIR_TEMPLATE = '../tmp/medium-xwwxsxv-wiki2/medium_xwwxsxv_wiki2_{sparsity}_eval'
# LOG_FILE = 'wiki2_xwwxsxv_eval_results.log'
# EVAL_METRICS = ['perplexity']

# 配置参数 wiki
# SHELL_FILE = 'run_clm.sh'
# MODEL_FILE = '../model_gpt2.py'
# OUTPUT_DIR_TEMPLATE = '../tmp/medium-ori-wiki2/medium_ori_wiki2_{sparsity}_eval'
# LOG_FILE = 'wiki2_ori_eval_results.log'
# EVAL_METRICS = ['perplexity']

# 配置参数squad
SHELL_FILE = 'run_qa.sh'
MODEL_FILE = '../model_gpt2.py'
OUTPUT_DIR_TEMPLATE = '../tmp/medium-ori-squad/medium_ori_squad_{sparsity}_eval'
LOG_FILE = 'squad_ori_eval_results.log'
EVAL_METRICS = ['eval_exact_match', 'eval_f1']  # 支持1-2个评估指标

# 稀疏度百分比列表
SPARSITY_LIST = [0,10,20,30,40,50,60,70,75,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99]

# 1. 更新 SHELLS 的输出目录
def update_shell(sparsity):
    with open(SHELL_FILE, 'r') as f:
        content = f.read()
    new_output_dir = OUTPUT_DIR_TEMPLATE.format(sparsity=sparsity)
    updated_content = re.sub(
        r'--output_dir\s+\S+',
        f'--output_dir {new_output_dir}',
        content
    )
    with open(SHELL_FILE, 'w') as f:
        f.write(updated_content)
    print(f"✅ 已更新 {SHELL_FILE} 的输出目录为 {new_output_dir}")

# 2. 更新 model_gpt2.py 的稀疏度阈值
def update_model(sparsity_float):
    with open(MODEL_FILE, 'r') as f:
        model_content = f.read()
    updated_model_content = re.sub(
        r'threshold\s*=\s*torch\.quantile\(inputx_kv\.abs\(\)\.flatten\(\),\s*0\.\d+\)',
        f'threshold = torch.quantile(inputx_kv.abs().flatten(), {sparsity_float:.2f})',
        model_content
    )
    with open(MODEL_FILE, 'w') as f:
        f.write(updated_model_content)
    print(f"✅ 已更新 {MODEL_FILE} 中的稀疏度为 {sparsity_float:.2f}")

# 3. 运行评估并提取指定的评估指标
def run_evaluation():
    print("🚀 开始运行评估...")
    result = subprocess.run(['bash', SHELL_FILE], capture_output=True, text=True)
    output = result.stdout + result.stderr
    
    # 提取指定的评估指标
    eval_results = {}
    for metric in EVAL_METRICS:
        if metric == 'perplexity':
            pattern = r'perplexity\s*=\s*(\d+\.\d+)'
        else:
            pattern = rf'{metric}\s*=\s*(\d+\.\d+)'
        
        match = re.search(pattern, output)
        if match:
            eval_results[metric] = match.group(1)
            print(f"📊 评估结果: {metric} = {match.group(1)}")
        else:
            eval_results[metric] = None
            print(f"⚠️ 无法从输出中提取 {metric}")
    
    print("✅ 评估完成")
    return output, eval_results

# 4. 主流程
if __name__ == "__main__":
    with open(LOG_FILE, 'w') as f:
        f.write(f"稀疏度批量评估结果 {SPARSITY_LIST}\n")
        f.write(f"评估指标: {', '.join(EVAL_METRICS)}\n")
        f.write("=" * 50 + "\n\n")
    
    for sparsity in SPARSITY_LIST:
        sparsity_float = sparsity / 100.0
        print(f"\n{'='*60}\n🔄 开始运行稀疏度 {sparsity}% 的评估\n{'='*60}\n")
        update_shell(sparsity)
        update_model(sparsity_float)
        start_time = time.time()
        output, eval_results = run_evaluation()
        end_time = time.time()
        
        with open(LOG_FILE, 'a') as f:
            f.write(f"稀疏度: {sparsity}%\n")
            f.write(f"运行时间: {end_time - start_time:.2f} 秒\n")
            
            # 写入所有评估指标的结果
            for metric in EVAL_METRICS:
                if eval_results[metric]:
                    f.write(f"{metric}: {eval_results[metric]}\n")
                else:
                    f.write(f"无法提取 {metric}\n")
            
            f.write("\n" + "-" * 30 + "\n\n")
        
        print(f"⏱️ 耗时: {end_time - start_time:.2f} 秒")
        print(f"📝 结果已保存到 {LOG_FILE}")
    
    print(f"\n{'='*60}\n🎉 所有评估已完成! 结果见 {LOG_FILE}")