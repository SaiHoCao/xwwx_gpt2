from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from accelerate.test_utils.testing import get_backend
from datasets import load_dataset,load_from_disk
from transformers.trainer_utils import get_last_checkpoint
import torch
from tqdm import tqdm

def calculate_perplexity(model_type="raw"):
    device, _, _ = get_backend()
    
    # 根据模型类型选择checkpoint
    if model_type == "raw":
        model_path = "../../gpt2"  # 未微调的原始GPT-2模型
    elif model_type == "default":
        last_checkpoint = get_last_checkpoint("./tmp/test-clm")  # 使用wikitext 微调，未改变gpt模型的chekpoint
        model_path = last_checkpoint
    elif model_type == "ori":
        last_checkpoint = get_last_checkpoint("./tmp/test-clm-ori")  # 使用wikitext 微调，改变gpt模型Attn部分的指定到具体的原注意力计算模式
        model_path = last_checkpoint
    elif model_type == "xwwx":
        last_checkpoint = get_last_checkpoint("./tmp/test-clm-xwwx")  # 使用wikitext 微调，改变gpt模型Attn部分的指定到xwwx注意力计算模式
        model_path = last_checkpoint
    else:
        raise ValueError("model_type must be one of: raw, default, ori, xwwx")

    print(f"\n正在加载 {model_type} 模型...")
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)

    # 加载测试数据
    print("正在加载测试数据...")
    raw = load_from_disk("../wikitext-2-raw-v1")
    test = raw["test"]

    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    
    print(f"开始计算 {model_type} 模型的困惑度...")
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        num_valid_tokens = (target_ids != -100).sum().item()
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens
    ppl = torch.exp(avg_nll)
    return ppl.item()

if __name__ == "__main__":
    # 计算所有模型类型的困惑度
    model_types = ["raw", "default", "ori", "xwwx"]
    results = {}
    
    for model_type in model_types:
        try:
            ppl = calculate_perplexity(model_type)
            results[model_type] = ppl
            print(f"\n{model_type} 模型在测试集上的困惑度(Perplexity): {ppl:.2f}")
        except Exception as e:
            print(f"\n计算 {model_type} 模型困惑度时出错: {str(e)}")
    
    print("\n=== 所有模型的困惑度对比 ===")
    for model_type, ppl in results.items():
        print(f"{model_type}: {ppl:.2f}")


# perplexity = evaluate.load("perplexity", module_type="metric")
# input_texts = datasets.load_dataset("wikitext",
#                                     "wikitext-2-raw-v1",
#                                     split="test")["text"][:50]
# input_texts = [s for s in input_texts if s!='']
# results = perplexity.compute(model_id='gpt2',
#                              predictions=input_texts)
# print(list(results.keys()))
# >>>['perplexities', 'mean_perplexity']
# print(round(results["mean_perplexity"], 2))
# >>>576.76
# print(round(results["perplexities"][0], 2))
# >>>889.28