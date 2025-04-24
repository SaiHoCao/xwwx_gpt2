from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

model_name = "../../gpt2"
# 加载模型配置
model = AutoModelForCausalLM.from_pretrained(model_name)

model.print_trainable_parameters()  

# 加载数据集
dataset = load_dataset("../wikitext-2-raw-v1")

print(dataset)

tokenizer = AutoTokenizer.from_pretrained(model_name)



config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="lora_only",
    modules_to_save=["decode_head"],
)
lora_model = get_peft_model(model, config)

model.print_trainable_parameters(lora_model)