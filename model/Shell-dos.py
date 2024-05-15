import argparse
import time

from gpt4all import GPT4All
import os

parser = argparse.ArgumentParser(description="演示如何使用argparse接收命令行参数")

# 添加模型参数
parser.add_argument("--model", default="mistral-7b-instruct-v0.1.Q4_0.gguf", type=str,)

# 解析参数
args = parser.parse_args()

current_path = './model'
model = args.model
print("path:", current_path)
print("model:", model)
try:
    model = GPT4All(model, model_path=current_path, allow_download=False,device='gpu')
except:
    print("Failed to load model")
    exit(-1)


# 计算generate_text的运行时间
def generate_text(_prompt, max_tokens=1024):
    output = model.generate(_prompt, max_tokens=max_tokens)
    return output

while True:
    prompt = input("prompt: ")
    start_time = time.time()
    output = generate_text(prompt)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", round(execution_time, 2), "seconds")
    print(output)
