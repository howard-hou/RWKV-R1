import re

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
                #pattern = re.compile(rf'boxed{{{re.escape(standard_answer)}}}')
                pattern = re.compile(f'{re.escape(standard_answer)}')
                if pattern.search(machine_answer):
                    correct_count += 1
    return correct_count

# 替换为你的实际文件路径
file_path = r'C:\Users\tyzha\Desktop\11\0218\220k-new\output_6.txt'
correct_lines = count_correct_lines(file_path)
print(f"正确的行数为: {correct_lines}")
