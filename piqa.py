import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import GPT2Tokenizer
from model_gpt2 import GPT2ForQuestionAnswering
from datasets import load_dataset,load_from_disk
from evaluate import evaluator

# 1. 初始化模型
last_checkpoint = get_last_checkpoint("./tmp/medium_squad_ori_10")

tokenizer = GPT2Tokenizer.from_pretrained(last_checkpoint)

model = GPT2ForQuestionAnswering.from_pretrained(last_checkpoint).eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# 2. 创建预测函数
def piqa_predictor(samples):
    predictions = []
    for goal, sol1, sol2 in zip(samples["goal"], samples["sol1"], samples["sol2"]):
        # 计算选项得分
        score1 = evaluate_qa(goal, sol1)
        score2 = evaluate_qa(goal, sol2)
        predictions.append(0 if score1 > score2 else 1)
    return predictions

# 3. QA评估函数
def evaluate_qa(question, candidate_answer):
    inputs = tokenizer(
        question,
        candidate_answer,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 使用联合概率作为相关性得分
    start_probs = torch.nn.functional.softmax(outputs.start_logits, dim=-1)
    end_probs = torch.nn.functional.softmax(outputs.end_logits, dim=-1)
    return (start_probs.max().item() + end_probs.max().item()) / 2

# 4. 专业评估
if __name__ == "__main__":
    raw_datasets = load_from_disk("../datasets/piqa")
    dataset = raw_datasets["validation"]
    # dataset = load_dataset("piqa", split="validation")
    # eval_module = evaluator("question-answering")
    
    # results = eval_module.compute(
    #     model_or_pipeline=piqa_predictor,
    #     data=dataset,
    #     metric=["accuracy", "f1"]
    # )
    
    # print("="*50)
    # print(f"零样本测试结果:")
    # print(f"准确率: {results['accuracy']:.2%}")
    # print(f"F1分数: {results['f1']:.2%}")
    # print("="*50)

    # 预测
    dataset = dataset.map(lambda x: {"prediction": piqa_predictor(x)})

    from evaluate import load
    accuracy = load("accuracy")
    f1 = load("f1")
    print("准确率:", accuracy.compute(predictions=dataset["prediction"], references=dataset["label"]))
    print("F1分数:", f1.compute(predictions=dataset["prediction"], references=dataset["label"]))