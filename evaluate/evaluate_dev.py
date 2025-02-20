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
from introduction import BASE_INTRO, R1_INTRO, BOX_INTRO, SHORT_INTRO

# 步骤 0: argparse
parser = argparse.ArgumentParser(description='Evaluate RWKV model on benchmarks')
parser.add_argument('model', type=str, help='path to model')
parser.add_argument('--dataset', type=str, default='HuggingFaceH4/MATH-500')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--intro', type=str, default='box', choices=['base', 'r1', 'box', 'short'])
parser.add_argument('--strategy', type=str, default='cuda fp16')
parser.add_argument('--output', type=str, default='output.txt')
parser.add_argument('--debug', action='store_true')

args = parser.parse_args()
if args.intro == 'base':
    INTRO = BASE_INTRO.strip()
    PATTERN = None
elif args.intro == 'r1':
    INTRO = R1_INTRO.strip()
    PATTERN = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
elif args.intro == 'box':
    INTRO = BOX_INTRO.strip()
    PATTERN = re.compile(r'\\boxed\{(.*?)\}', re.DOTALL)
elif args.intro == 'short':
    INTRO = SHORT_INTRO.strip()
    PATTERN = None
else:
    raise ValueError("Invalid intro type")

print(f"Using intro type: {args.intro}")
print(INTRO)
print(PATTERN)

# 步骤 1：加载模型和分词器
model_path = Path(args.model).parent / Path(args.model).stem
model = RWKV(model=str(model_path), strategy=args.strategy)
# 初始化分词器管道
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
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

def postprocess_answer(model_answer, pattern):
    # 这里可以添加后处理逻辑，如去掉多余空格等
    model_answer = model_answer.strip()
    # extract <answer> ans </answer>
    # Compile the regular expression with re.DOTALL to match across lines
    if pattern is not None:
        match = pattern.search(model_answer)
        if match:
            model_answer = match.group(1).strip()
        else: # format fail
            model_answer = None
    else:
        model_answer = model_answer
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
    
    prompt = f"{INTRO}\n\nUser: {question}\n\nAssistant:"
    raw_answer = pipeline.generate(prompt, token_count=10, args=pipe_args)
    model_answer = postprocess_answer(raw_answer, PATTERN)
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
  
