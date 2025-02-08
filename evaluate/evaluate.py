import os

os.environ["RWKV_V7_ON"] = '1' # ==> enable RWKV-7 mode
os.environ['RWKV_JIT_ON'] = '1' # '1' for better speed
os.environ["RWKV_CUDA_ON"] = '0' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

import torch
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
import json

# 步骤 1：加载模型和分词器
model = RWKV(model='/path/to/model', strategy='cuda fp16')
# 初始化分词器管道
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

# 步骤 2：读取数据集
math500_data = []
with open('test.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        try:
            data = json.loads(line)
            question = data.get('problem')
            answer = data.get('answer')
            if question and answer:
                math500_data.append({"question": question, "answer": answer})
        except json.JSONDecodeError:
            print(f"Error decoding JSON in line: {line}")

# 步骤 3：定义评估函数
def evaluate_answer(model_answer, correct_answer):
    # 这里可以添加更复杂的答案评估逻辑，如处理格式不一致等问题
    return model_answer.strip() == correct_answer.strip()

# 步骤 4：使用模型回答问题并评估
correct_count = 0
total_count = len(math500_data)

item_count = 0

with open('output_base1.5.txt', 'w', encoding='utf-8') as file:

    for data in math500_data:
        question = data["question"]
        correct_answer = data["answer"]

        # 使用模型生成回答
        args = PIPELINE_ARGS(temperature=1.0, top_p=0.85, top_k=0)
        
        question = "Return your final response within \\boxed{}." + question
        
        #question += "After completing your logical reasoning and calculations for this problem, kindly conclude your response with 【ANSWER: 】 followed by the final result. Ensure that all the steps in your reasoning are clear and logical, and refrain from including any extraneous information in the final answer section."
        
        model_answer = pipeline.generate(question, token_count=4096, args=args).replace("\n", "")
        
        item_count += 1
        file.write(f"{item_count}\t【{correct_answer}】\t{model_answer}\n")
        #print(model_answer," ++++ ", correct_answer, "\n")
        
        print(item_count)

# 步骤 5：统计准确率
def count_correct_lines(file_path):
    correct_count = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 按制表符分割每行内容
            parts = line.strip().split('\t')
            if len(parts) == 3:
                # 提取第二个内容（标准答案）并去掉方括号
                standard_answer = parts[1].strip('【】')
                # 提取第三个内容（机器生成答案）
                machine_answer = parts[2]
                # 构建正则表达式模式，用于检查机器生成答案中是否包含 boxed{标准答案}
                pattern = re.compile(rf'boxed{{{re.escape(standard_answer)}}}')
                if pattern.search(machine_answer):
                    correct_count += 1
    return correct_count

# 替换为你的实际文件路径
file_path = "output_base1.5.txt'
correct_lines = count_correct_lines(file_path)
print(f"正确的行数为: {correct_lines}")
  
