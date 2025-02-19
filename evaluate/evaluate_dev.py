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
from datasets import load_dataset, load_from_disk

import re
import argparse
from prompt import BASE_INTRO, R1_INTRO

# 步骤 0: argparse
parser = argparse.ArgumentParser(description='Evaluate RWKV model on benchmarks')
parser.add_argument('model', type=str, help='path to model')
parser.add_argument('--dataset', type=str, default='HuggingFaceH4/MATH-500')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--strategy', type=str, default='cuda fp16')
parser.add_argument('--output', type=str, default='output.txt')

args = parser.parse_args()

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

# 步骤 4：使用模型回答问题并评估
correct_count = 0
total_count = len(benchmark_data)

item_count = 0

for line in tqdm(benchmark_data):
    question = line["question"]
    correct_answer = line["answer"]
    
    # question = "Return your final response within \\boxed{}." + question
    # prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the userwith the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>\n<answer> answer here </answer>. User: " + question + " Assistant: "

    
    question = "Write $\\frac{3}{20}$ as a decimal."
    prompt = f"{R1_INTRO}\n\nUser: {question}\n\nAssistant:"
    model_answer = pipeline.generate(prompt, token_count=512, args=pipe_args)
    
    item_count += 1
    print("prompt:", prompt)
    print(f"Model Answer:{model_answer}")
    print("Model Answer Length:", len(model_answer))
    #file.write(f"{item_count}\t【{correct_answer}】\t{model_answer}\n")
    # print(f"{item_count}\t{correct_answer}\t{model_answer}\n")
    # print(model_answer," ++++ ", correct_answer, "\n")
    
    # print(item_count)
    break

# # 步骤 5：统计准确率
# def count_correct_lines(file_path):
#     correct_count = 0
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             # 按制表符分割每行内容
#             parts = line.strip().split('\t')
#             if len(parts) == 3:
#                 # 提取第二个内容（标准答案）并去掉方括号
#                 standard_answer = parts[1].strip('【】')
#                 # 提取第三个内容（机器生成答案）
#                 machine_answer = parts[2]
#                 # 构建正则表达式模式，用于检查机器生成答案中是否包含 boxed{标准答案}
#                 pattern = re.compile(rf'boxed{{{re.escape(standard_answer)}}}')
#                 if pattern.search(machine_answer):
#                     correct_count += 1
#     return correct_count

# # 替换为你的实际文件路径
# file_path = "output_base1.5.txt'
# correct_lines = count_correct_lines(file_path)
# print(f"正确的行数为: {correct_lines}")
  
