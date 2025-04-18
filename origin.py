from transformers import GPT2Model

model = GPT2Model.from_pretrained("/home/csh/data/gpt2")

print(model)
# 打印模型配置

print(model.config.reorder_and_upcast_attn)
print(model.config)