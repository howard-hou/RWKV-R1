import os

os.environ["RWKV_V7_ON"] = '1' # ==> enable RWKV-7 mode
os.environ['RWKV_JIT_ON'] = '1' # '1' for better speed
os.environ["RWKV_CUDA_ON"] = '0' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

import torch
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

import re
import argparse
from prompt import S1_INTRO

# 步骤 0: argparse
parser = argparse.ArgumentParser(description='Evaluate RWKV model on benchmarks')
parser.add_argument('model', type=str, help='path to model')
parser.add_argument('--dataset', type=str, default='HuggingFaceH4/MATH-500')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--strategy', type=str, default='cuda fp16')
parser.add_argument('--output', type=str, default='output.txt')
parser.add_argument('--debug', action='store_true')

args = parser.parse_args()

# 步骤 1：加载模型和分词器
model_path = Path(args.model).parent / Path(args.model).stem
model = RWKV(model=str(model_path), strategy=args.strategy)
# 初始化分词器管道
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
# stop_token_ids = pipeline.encode("</think>")
# print(stop_token_ids)
# exit()
pipe_args = PIPELINE_ARGS(temperature = 0.0, top_p = 0.5, top_k = 0, # top_k = 0 then ignore
                     alpha_frequency = 0.5,
                     alpha_presence = 0.5,
                     alpha_decay = 0.996, # gradually decay the penalty
                     token_ban = [], # ban the generation of some tokens
                     token_stop = [261], # stop generation whenever you see any token here
                     chunk_len = 1024) # split input into chunks to save VRAM (shorter -> slower)

# 步骤 2：读取数据集
benchmark_data = []
ds = load_dataset(args.dataset)
if args.split in ds:
    ds = ds[args.split]
print(f"Loaded {len(ds)} examples from {args.dataset}/{args.split}")

for line in ds:
    question = line.get('problem')
    answer = line.get('answer')
    if question and answer:
        benchmark_data.append({"question": question, "answer": answer})
print(f"Loaded {len(benchmark_data)} examples with both question and answer")

# 步骤 3：定义评估函数
def evaluate_answer(model_answer, correct_answer):
    # 这里可以添加更复杂的答案评估逻辑，如处理格式不一致等问题
    return model_answer.strip() == correct_answer.strip()

def postprocess_answer(model_answer):
    # 这里可以添加后处理逻辑，如去掉多余空格等
    model_answer = model_answer.strip()
    return model_answer

# 步骤 4：使用模型回答问题并评估
if args.debug:
    benchmark_data = benchmark_data[:10] # for debug
total_count = len(benchmark_data)
ans_fail_case = []
format_fail_case = []
for line in tqdm(benchmark_data, desc="Evaluating"):
    question = line["question"]
    correct_answer = line["answer"]
    
    prompt = f"{S1_INTRO}\n\nUser: {question}\n\nAssistant: <think>"
    think_step = pipeline.generate(prompt, token_count=100, args=pipe_args)
    if "</think>" in think_step:
        think_step = think_step.split("</think>")[0]
    think_step += " </think>\nFinal Answer:"
    prompt_with_think = prompt + think_step
    raw_answer = pipeline.generate(prompt_with_think, token_count=30, args=pipe_args)
    model_answer = postprocess_answer(raw_answer)
    if model_answer is None:
        format_fail_case.append((question, raw_answer, correct_answer))
        continue
    if not evaluate_answer(model_answer, correct_answer):
        ans_fail_case.append((question, model_answer, correct_answer))


format_correct_count = total_count - len(format_fail_case)
format_fail_count = len(format_fail_case)
ans_correct_count = format_correct_count - len(ans_fail_case)
print(f"Total examples: {total_count}")
print(f"Correct format: {format_correct_count}")
print(f"Fail format: {format_fail_count}")
print(f"Correct answers: {ans_correct_count}")
print(f"Format accuracy: {format_correct_count / total_count:.2f}")
print(f"Answer accuracy: {ans_correct_count / format_correct_count:.2f}")

# 步骤 5：print bad cases
if format_fail_case and args.debug:
    print("\nFormat Fail Cases:")
    for i, (question, model_answer, correct_answer) in enumerate(format_fail_case):
        print(f"Case {i+1}:")
        print(f"Question: {question}")
        print(f"Model Answer: {model_answer}")
        print(f"Correct Answer: {correct_answer}")
        print()
if ans_fail_case and args.debug:
    print("\nAnswer Fail Cases:")
    for i, (question, model_answer, correct_answer) in enumerate(ans_fail_case):
        print(f"Case {i+1}:")
        print(f"Question: {question}")
        print(f"Model Answer: {model_answer}")
        print(f"Correct Answer: {correct_answer}")
        print()
  
