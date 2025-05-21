from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.trainer_utils import get_last_checkpoint
import torch

def generate_text(prompt, model_type="default", max_length=30):
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

    # 加载模型和分词器
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    # 将模型设置为评估模式
    model.eval()
    
    # 对输入文本进行编码
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones_like(inputs)
    
    # 生成文本
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            output_hidden_states=True,  # 确保输出hidden_states
            return_dict_in_generate=True  # 返回字典格式
        )
    
    # 获取hidden_states
    hidden_states = outputs.hidden_states  # 这是一个元组，包含每一层的hidden_states
    
    # 保存hidden_states
    torch.save(hidden_states, f'hidden_states_{model_type}.pt')
    
    # 解码生成的文本
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    # 打印hidden_states的形状信息
    # print(f"\nHidden States信息:")
    # for i, layer_states in enumerate(hidden_states):
    #     # 每个layer_states是一个元组，包含每个时间步的hidden_state
    #     print(f"第{i}层 hidden_states:")
    #     for t, state in enumerate(layer_states):
    #         print(f"  时间步{t} shape: {state.shape}")
    
    return generated_text, hidden_states

if __name__ == "__main__":
    # 测试提示
    prompt = "Hello, I'm a student,"
    
    # 测试所有模型
    print("=== 测试未微调的原始GPT-2模型 ===")
    generated_raw = generate_text(prompt, model_type="raw")
    print(f"输入: {prompt}")
    print(f"生成: {generated_raw}\n")
    
    print("=== 测试微调但未改变的GPT-2模型 ===")
    generated_default = generate_text(prompt, model_type="default")
    print(f"输入: {prompt}")
    print(f"生成: {generated_default}\n")
    
    print("=== 测试改变模型定向到原始注意力计算模式 ===")
    generated_ori = generate_text(prompt, model_type="ori")
    print(f"输入: {prompt}")
    print(f"生成: {generated_ori}\n")
    
    print("=== 测试改变模型定向到XWWX注意力计算模式 ===")
    generated_xwwx = generate_text(prompt, model_type="xwwx")
    print(f"输入: {prompt}")
    print(f"生成: {generated_xwwx}")
