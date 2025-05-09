from transformers import GPT2Tokenizer, GPT2Model
import torch

def get_text_representation(model, tokenizer, input_text):
    # 对输入文本进行编码
    encoded_input = tokenizer(input_text, return_tensors='pt', truncation=True)
    
    # 获取模型输出
    with torch.no_grad():
        output = model(**encoded_input)
    
    # 获取最后一层的隐藏状态
    last_hidden_states = output.last_hidden_state
    
    return last_hidden_states

# 加载模型和分词器
model = GPT2Model.from_pretrained("../../gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("../../gpt2")
model.eval()

# 使用示例
if __name__ == "__main__":
    input_text = "今天天气真好，"
    hidden_states = get_text_representation(model, tokenizer, input_text)
    print(f"输入文本: {input_text}")
    print(f"隐藏状态形状: {hidden_states.shape}")
    print(f"隐藏状态示例: {hidden_states[0, 0, :5]}")  # 打印第一个token的前5个维度