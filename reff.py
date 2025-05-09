from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.trainer_utils import get_last_checkpoint
import torch

def generate_text(prompt, max_length=30):

    last_checkpoint = get_last_checkpoint("./tmp/test-clm") # 使用wikitext 微调，未改变gpt模型的chekpoint
    last_checkpoint = get_last_checkpoint("./tmp/test-clm-ori") # 使用wikitext 微调，改变gpt模型Attn部分的
    last_checkpoint = get_last_checkpoint("./tmp/test-clm-xwwx")
    # 加载模型和分词器
    # model_path = "../../gpt2"  # 未微调的huggingface gpt2权重
    model_path = last_checkpoint
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    # 将模型设置为评估模式
    model.eval()
    
    # 对输入文本进行编码
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones_like(inputs)  # 添加attention mask
    
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
        )
    
    # 解码生成的文本
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    # 测试生成
    prompt = "Hello, I'm a student,"
    generated = generate_text(prompt)
    print(f"输入: {prompt}")
    print(f"生成: {generated}")
