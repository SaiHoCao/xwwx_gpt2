# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import torch

# # 加载模型和分词器
# tokenizer = GPT2Tokenizer.from_pretrained("../../gpt2")
# model = GPT2LMHeadModel.from_pretrained("../../gpt2")
# # model.load_state_dict(torch.load("path/to/checkpoint.pth"))
# model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

# # 输入句子
# input_text = "人工智能的未来是"

# # 分词和预处理
# inputs = tokenizer.encode_plus(
#     input_text,
#     return_tensors="pt",
#     add_special_tokens=True,
#     max_length=64,
#     truncation=True,
# )
# inputs = {k: v.to(model.device) for k, v in inputs.items()}

# # 生成文本
# outputs = model.generate(
#     **inputs,
#     max_length=100,
#     do_sample=True,
#     temperature=0.7,
#     top_p=0.92,
#     num_return_sequences=3,  # 生成 3 个候选结果
# )

# # 打印所有生成结果
# for i, output in enumerate(outputs):
#     text = tokenizer.decode(output, skip_special_tokens=True)
#     print(f"结果 {i+1}: {text}\n")

# from transformers import GPT2LMHeadModel, GPT2Tokenizer

# # 1. 下载并加载模型和分词器
# model_name = "../../gpt2"  # 可以是 "gpt2-medium", "gpt2-large" 等变体
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPT2LMHeadModel.from_pretrained(model_name)

# # 2. 将模型设置为评估模式（关闭dropout等训练特性）
# model.eval()

# # 3. 输入文本处理
# prompt_text = "Hello, I'm a language model,"
# input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

# # 4. 生成文本（推理）
# output = model.generate(
#     input_ids,
#     max_length=100,  # 生成文本的最大长度
#     num_return_sequences=1,  # 生成多少个候选结果
#     temperature=0.9,  # 控制随机性 (较低值更保守，较高值更随机)
#     top_k=50,  # 限制最高概率的token数量
#     top_p=0.95,  # nucleus sampling参数
#     repetition_penalty=1.2,  # 防止重复
#     pad_token_id=tokenizer.eos_token_id  # 设置结束符
# )

# # 5. 解码并打印结果
# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
# print("Generated Text:\n", generated_text)

from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('../../gpt2')
model = GPT2Model.from_pretrained('../../gpt2')
text = "Hello, I'm a language model,"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

