
import os

os.environ["RWKV_V7_ON"] = '1' # ==> enable RWKV-7 mode
os.environ['RWKV_JIT_ON'] = '1' # '1' for better speed
os.environ["RWKV_CUDA_ON"] = '1' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
import json
from introduction import R1_INTRO
import re

# 步骤 1：加载模型和分词器
#model = RWKV(model='/data/Tianyu/RWKV-R1/models/RWKV-x070-World-1.5B-v3-20250127-ctx4096', strategy='cuda fp16')
model = RWKV(model='/data/Ryan/models/rwkv1b5-sft/20250219_220k_think_answer/rwkv1b5-sft/rwkv-2', strategy='cuda fp16')
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
item_count = 0
total_count = len(math500_data)

with open('output.txt', 'w', encoding='utf-8') as file:

    for data in math500_data:
        question = data["question"]
        correct_answer = data["answer"]

        # 使用模型生成回答
        pipe_args = PIPELINE_ARGS(temperature = 0.0, top_p = 0.5, top_k = 0, # top_k = 0 then ignore
                     alpha_frequency = 0.5,
                     alpha_presence = 0.5,
                     alpha_decay = 0.996, # gradually decay the penalty
                     token_ban = [], # ban the generation of some tokens
                     token_stop = [261], # stop generation whenever you see any token here
                     chunk_len = 1024) # split input into chunks to save VRAM (shorter -> slower)
        
        #question = "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop a well-considered thinking process. Please structure your response into a thinking section.In the thinking section, detail your reasoning process using the specified format:                 <think>                 {thought with steps separated with '\n\n'}                 </think>             Each step should include detailed considerations such as analyzing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps.              After the thinking section, directly present the final solution in a precise and clear format without additional labels." + "Return your final response within \\boxed{}." + question
        
        #question += "After completing your logical reasoning and calculations for this problem, kindly conclude your response with 【ANSWER: 】 followed by the final result. Ensure that all the steps in your reasoning are clear and logical, and refrain from including any extraneous information in the final answer section."
        
        prompt = f"{R1_INTRO}\n\nUser: {question}\n\nAssistant:"
        model_answer = pipeline.generate(prompt, token_count=1024, args=pipe_args).replace("\n", "\\n")
        
        item_count += 1
        file.write(f"{item_count}\t【{correct_answer}】\t{model_answer}\n")
        file.flush()
        #print(model_answer," ++++ ", correct_answer, "\n")
        
        print(item_count)

# 统计准确率
correct_count = 0
format_correct_count = 0
with open('output.txt', 'r', encoding='utf-8') as file:
    for line in file:
        # 按制表符分割每行内容
        parts = line.strip().split('\t')
        if len(parts) == 3:
            # 提取第二个内容（标准答案）并去掉方括号
            standard_answer = parts[1].strip('【】')
            # 提取第三个内容（机器生成答案）
            machine_answer = parts[2]

            # 统计格式正确率
            think_start = "<think>"
            think_end = "</think>"
            answer_start = "<answer>"
            answer_end = "</answer>"
            text = machine_answer
            # 检查四种标签是否都存在于文本中
            if think_start in text and think_end in text and answer_start in text and answer_end in text:
                format_correct_count += 1
                    
            pattern = re.compile(f'{re.escape(standard_answer)}')
            if pattern.search(machine_answer):
                correct_count += 1


print(f"格式正确的行数为: {format_correct_count}")
print(f"答案正确的行数为: {correct_count}")
print(f"格式准确率为: {format_correct_count / total_count * 100:.2f}%")
print(f"答案准确率为: {correct_count / total_count * 100:.2f}%")
