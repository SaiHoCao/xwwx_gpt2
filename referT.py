from transformers import GPT2Tokenizer
from model_gpt2 import GPT2LMHeadModel
import torch
from deepspeed.profiling.flops_profiler import FlopsProfiler
from transformers.trainer_utils import get_last_checkpoint
import time
# from deepspeed.profiling.flops_profiler import get_model_profile

def get_model_profile(model,
                      args=[],
                      kwargs={},
                      print_profile=True,
                      detailed=True,
                      module_depth=-1,
                      top_modules=1,
                      warm_up=1,
                      as_string=True,
                      output_file=None,
                      ignore_modules=None,
                      mode='forward'):

    prof = FlopsProfiler(model)

    for _ in range(warm_up):
        if mode == 'forward':
            _ = model(*args, **kwargs)
        if mode == 'generate':
            _ = model.generate(*args, **kwargs)

    prof.start_profile(ignore_list=ignore_modules)

    if mode == 'forward':
        with torch.no_grad():   
            _ = model(*args, **kwargs)
    if mode == 'generate':
        with torch.no_grad():
            _ = model.generate(*args, **kwargs)


    flops = prof.get_total_flops(as_string)
    macs = prof.get_total_macs(as_string)
    params = prof.get_total_params(as_string)
    duration = prof.get_total_duration(as_string)
    if print_profile:
        prof.print_model_profile(profile_step=warm_up,
                                 module_depth=module_depth,
                                 top_modules=top_modules,
                                 detailed=detailed,
                                 output_file=output_file)

    prof.end_profile()
    return flops, macs, params, duration

def setup_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    last_checkpoint = get_last_checkpoint("./tmp/test-clm-medium")
    tokenizer = GPT2Tokenizer.from_pretrained(last_checkpoint)
    
    model = GPT2LMHeadModel.from_pretrained(last_checkpoint, use_cache=True).to(device)
    # 转换为bf16
    # model = model.to(torch.bfloat16)
    model = model.half()
    # 只保留第一层
    model.transformer.h = model.transformer.h[:1]
    model.n_layers = 1

    model.eval()
    # print(f"model.config: {model.config}")
    # 检查第一个参数的数据类型
    # first_param_dtype = next(model.parameters()).dtype
    # print(f"模型参数类型: {first_param_dtype}")
    # print(f"max_position_embeddings: {model.config.max_position_embeddings}")
    # print(model)
    # print(f"model.n_layers: {model.n_layers}")
    return model, tokenizer, device


def run_inference(model, seq_len, use_cache=False):
    # 1. 全量推理（use_cache=False）
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_len)).to(device)
    print(f"\n=== 全量推理（use_cache=False）- 序列长度: {seq_len} ===")

    # warmup
    with torch.no_grad():
        outputs = model(input_ids, use_cache=use_cache)
    
    profiler = FlopsProfiler(model)
    profiler.start_profile()
    start = time.time()
    with torch.no_grad():
        outputs = model(input_ids, use_cache=use_cache)
    end = time.time()
    profiler.stop_profile()
    print(f"FLOPs: {profiler.get_total_flops(as_string=True)}")
    print(f"MACs: {profiler.get_total_macs(as_string=True)}")
    print(f"Parameters: {profiler.get_total_params(as_string=True)}")
    print(f"Latency: {profiler.get_total_duration(as_string=True)}")
    print(f"Time: {end - start:.6f} 秒")
    profiler.end_profile()
    # start = time.time()
    # flops, macs, params, duration = get_model_profile(
    #     model,
    #     args=(input_ids,),
    #     kwargs={'mode': 'forward', 'use_cache': use_cache, 'past_key_values': None},
    #     print_profile=False,
    # )
    # end = time.time()
    # print(f"FLOPs: {flops}")
    # print(f"MACs: {macs}")
    # print(f"Params: {params}")
    # print(f"Duration: {duration}")
    # print(f"Time: {end - start:.6f} 秒")
    return

def run_autoregressive_inference(model, seq_len, use_cache=True):
    # 2. 自回归推理（use_cache=True）
    # 确保序列长度不超过模型的最大长度限制
    max_length = model.config.max_position_embeddings
    if seq_len - 1 >= max_length:
        print(f"Warning: sequence length {seq_len} exceeds model max length {max_length}, using {max_length-1}")
        seq_len = max_length - 1  # 留出一个位置给单步推理
    
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_len - 1)).to(device)
    print(f"\n=== 自回归推理（use_cache=True）- 序列长度: {seq_len} ===")
    
    # 2.1 先准备历史缓存
    with torch.no_grad():
        past = model(input_ids, use_cache=use_cache).past_key_values

    # 2.2 单步推理
    next_token = torch.randint(0, tokenizer.vocab_size, (1, 1)).to(device)

    # warmup
    with torch.no_grad():
        outputs = model(next_token, past_key_values=past, use_cache=use_cache)
    
    profiler = FlopsProfiler(model)
    profiler.start_profile()
    start = time.time()
    with torch.no_grad():
        outputs = model(next_token, past_key_values=past, use_cache=use_cache)
    end = time.time()
    profiler.stop_profile()
    print(f"FLOPs: {profiler.get_total_flops(as_string=True)}")
    print(f"MACs: {profiler.get_total_macs(as_string=True)}")
    print(f"Parameters: {profiler.get_total_params(as_string=True)}")
    print(f"Latency: {profiler.get_total_duration(as_string=True)}")
    print(f"Time: {end - start:.6f} 秒")
    profiler.end_profile()
    # profiler.print_model_profile()
    # start = time.time()
    # flops, macs, params, duration = get_model_profile(
    #     model,
    #     args=(next_token,),
    #     kwargs={'mode': 'forward', 'use_cache': use_cache, 'past_key_values': past},
    #     print_profile=False,
    # )
    # end = time.time()
    # print(f"FLOPs: {flops}")
    # print(f"MACs: {macs}")
    # print(f"Params: {params}")
    # print(f"Duration: {duration}")
    # print(f"Time: {end - start:.6f} 秒")
    return


if __name__ == "__main__":
    # 初始化模型
    model, tokenizer, device = setup_model()
    
    # run_autoregressive_inference(model, 1024, use_cache=True)
    # run_inference(model, 1024, use_cache=False)

    # run_autoregressive_inference(model, 512, use_cache=True)
    run_inference(model, 512, use_cache=False)