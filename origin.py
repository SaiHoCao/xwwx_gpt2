from transformers.models.gpt2 import modeling_gpt2
from transformers import GPT2Model
from model import GPT2AttentionXWWX


# model.GPT2Attention = GPT2AttentionXWWX
modeling_gpt2.GPT2Attention = GPT2AttentionXWWX
model = GPT2Model.from_pretrained("../../gpt2")
model.config.use_cache = False  
model.config._attn_implementation = "eager"
model.config.reorder_and_upcast_attn = True

print(model)
# 打印模型配置

print(model.config.use_cache)
print(model.config._attn_implementation)
print(model.config.reorder_and_upcast_attn)
print(model.config)