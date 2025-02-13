"""
Torch-NPU 版本的 RWKV 推理代码
参考自 RWKV 官方代码，已将模型加载、推理和 Lambada 测试部分修改为使用 NPU，
并在代码开始时初始化 ACL、设置设备，在结束时释放设备资源。
"""

import os
import types, gc, math, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# 导入 torch_npu 以及相关工具
import torch_npu
#from torch_npu.contrib import transfer_to_npu  # 如果需要也可用此接口做 tensor 转移
import acl

# 初始化 ACL 并设置 NPU 设备（设备索引 0）
acl.init()
acl.rt.set_device(6)
# 注意：在使用 NPU 时，device 字符串通常为 "npu"
device = torch.device("npu")

np.set_printoptions(precision=4, suppress=True, linewidth=200)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch._C._jit_set_autocast_mode(False)

################################################################################
# 模型参数设置
################################################################################

args = types.SimpleNamespace()

# 模型文件下载地址：https://huggingface.co/BlinkDL/rwkv-7-pile
# 注意：请确保 MODEL_PATH 指向正确的模型文件（例如 168M 或 421M 模型）
#MODEL_PATH = "../../models/RWKV-x070-Pile-168M-20241120-ctx4096.pth"
#MODEL_PATH = "/mnt/program/RWKV-x070-Pile-421M-20241127-ctx4096.pth"
MODEL_PATH = "../../models/RWKV-x070-World-1.5B-v3-20250127-ctx4096-self.pth"

if '168M' in MODEL_PATH:
    args.n_layer = 12
    args.n_embd = 768
    D_DECAY_LORA = 64
    D_AAA_LORA = 64
    D_MV_LORA = 32
    D_GATE_LORA = 128
    args.vocab_size = 50304  # “pile”模型原始词表为50277，padding 到50304
elif '421M' in MODEL_PATH:
    args.n_layer = 24
    args.n_embd = 1024
    D_DECAY_LORA = 64
    D_AAA_LORA = 64
    D_MV_LORA = 64
    D_GATE_LORA = 128
    args.vocab_size = 50304  # “pile”模型原始词表为50277，padding 到50304
elif '1.5B' in MODEL_PATH:
    args.n_layer = 24
    args.n_embd = 2048
    D_DECAY_LORA = 96
    D_AAA_LORA = 96
    D_MV_LORA = 64
    D_GATE_LORA = 256
    args.vocab_size = 65536  # “pile”模型原始词表为50277，padding 到50304


# 使用 tokenizers 库加载分词器
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("../RWKV-v4neo/20B_tokenizer.json")

# 推荐使用半精度（half）以加快推理速度
DTYPE = torch.half

args.head_size_a = 64  # 固定参数，不要更改
HEAD_SIZE = args.head_size_a

# 是否使用 CUDA kernel 优化（此处关闭，因为我们用的是 NPU，直接用普通实现）
USE_CUDA_KERNEL = False

# 为方便将 torch.jit.script 方法替换为 __nop
def __nop(ob):
    return ob

MyModule = nn.Module
MyFunction = __nop
# 如果需要使用 TorchScript 优化，可以参考 RWKV 官方示例

################################################################################
# RWKV7_OP 实现（这里走非 CUDA kernel 分支，直接使用 Python for 循环）
################################################################################

def RWKV7_OP(r, w, k, v, a, b):
    # 输入尺寸均为 [B, T, C]，内部会 reshape 为 [B, T, H, N]，其中 H = C // HEAD_SIZE, N = HEAD_SIZE
    B, T, C = r.size()
    H = C // HEAD_SIZE
    N = HEAD_SIZE
    # 转换为 float（后面运算中累加误差较小时，最后转换回 half）
    r = r.view(B, T, H, N).half()
    k = k.view(B, T, H, N).half()
    v = v.view(B, T, H, N).half()
    a = a.view(B, T, H, N).half()
    b = b.view(B, T, H, N).half()
    # 这里对 w 做指数运算，类似于“软衰减”
    w = torch.exp(-torch.exp(w.view(B, T, H, N).half()))
    out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)
    state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)


#因为优化了这个点tokens推理快了10多倍
    # for t in range(T):   #这部分循环在昇腾上太费时间了，改成矩阵的形式就可以了  这个是优化点
    #     kk = k[:, t, :].view(B, H, 1, N)
    #     rr = r[:, t, :].view(B, H, N, 1)
    #     vv = v[:, t, :].view(B, H, N, 1)
    #     aa = a[:, t, :].view(B, H, N, 1)
    #     bb = b[:, t, :].view(B, H, 1, N)
    #     state = state * w[:, t, :, None, :] + state @ aa @ bb + vv @ kk
    #     out[:, t, :] = (state @ rr).view(B, H, N)
    kk = k.view(B, T, H, 1, N)
    rr = r.view(B, T, H, N, 1)
    vv = v.view(B, T, H, N, 1)
    aa = a.view(B, T, H, N, 1)
    bb = b.view(B, T, H, 1, N)
    w_expanded = w.unsqueeze(3)  # [B, T, H, 1, N]

    # 向量化计算 RWKV7_OP
    state = state * w_expanded + torch.matmul(state, torch.matmul(aa, bb)) + torch.matmul(vv, kk)
    out = torch.matmul(state, rr).view(B, T, C)


    return out.view(B, T, C).to(dtype=DTYPE)

################################################################################
# RWKV 模型各模块定义
################################################################################

# RWKV TimeMix 层
class RWKV_Tmix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        H = self.n_head
        N = self.head_size
        C = args.n_embd

        self.x_r = nn.Parameter(torch.empty(1, 1, C))
        self.x_w = nn.Parameter(torch.empty(1, 1, C))
        self.x_k = nn.Parameter(torch.empty(1, 1, C))
        self.x_v = nn.Parameter(torch.empty(1, 1, C))
        self.x_a = nn.Parameter(torch.empty(1, 1, C))
        self.x_g = nn.Parameter(torch.empty(1, 1, C))

        self.w0 = nn.Parameter(torch.empty(1, 1, C))
        self.w1 = nn.Parameter(torch.empty(C, D_DECAY_LORA))
        self.w2 = nn.Parameter(torch.empty(D_DECAY_LORA, C))

        self.a0 = nn.Parameter(torch.empty(1, 1, C))
        self.a1 = nn.Parameter(torch.empty(C, D_AAA_LORA))
        self.a2 = nn.Parameter(torch.empty(D_AAA_LORA, C))

        self.v0 = nn.Parameter(torch.empty(1, 1, C))
        self.v1 = nn.Parameter(torch.empty(C, D_MV_LORA))
        self.v2 = nn.Parameter(torch.empty(D_MV_LORA, C))

        self.g1 = nn.Parameter(torch.empty(C, D_GATE_LORA))
        self.g2 = nn.Parameter(torch.empty(D_GATE_LORA, C))

        self.k_k = nn.Parameter(torch.empty(1, 1, C))
        self.k_a = nn.Parameter(torch.empty(1, 1, C))
        self.r_k = nn.Parameter(torch.empty(H, N))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)  # 注意 eps 值

    @MyFunction
    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5  # 限制 w 的范围
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v  # 保存第一层的 v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        x_out = RWKV7_OP(r, w, k, v, -kk, kk * a)
        x_out = self.ln_x(x_out.view(B * T, C)).view(B, T, C)
        
        x_out = x_out + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        x_out = self.output(x_out * g)
        return x_out, v_first

# RWKV ChannelMix 层
class RWKV_CMix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            self.x_k = nn.Parameter(torch.empty(1, 1, args.n_embd))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)

# RWKV Block 层
class Block(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln0 = nn.LayerNorm(args.n_embd)  # 仅在第一层使用
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)
        
    @MyFunction
    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)
        xx, v_first = self.att(self.ln1(x), v_first)
        x = x + xx
        x = x + self.ffn(self.ln2(x))
        return x, v_first

# 整个 RWKV 模型
class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        args.dim_att = args.n_embd
        args.dim_ffn = args.n_embd * 4
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def forward(self, idx):
        x = self.emb(idx)
        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)
        x = self.ln_out(x)
        x = self.head(x)
        return x

################################################################################
# 模型加载与推理
################################################################################

# 加载模型参数（先在 CPU 上加载，再送入 NPU）
model_params = torch.load(MODEL_PATH, map_location="cpu")

with torch.no_grad():
    # 构造模型，送入 NPU，并转换为半精度
    model = RWKV(args).to(device)
    model = model.half()
    model.load_state_dict(model_params, strict=False)  # 注意：这里忽略部分未加载的参数

    ############################################################################
    # 示例：给定一句话，预测下一个 token 的概率分布
    ############################################################################
    prompt = "The Eiffel tower is in the city of"
    input_ids = tokenizer.encode(prompt).ids
    print(f'\nInput token ids:\n{input_ids}')

    # 注意：构造 tensor 时指定 device 为 NPU
    input_tensor = torch.tensor(input_ids, device=device).reshape(1, -1)

    start_time = time.time()  # 记录开始时间
    out = model(input_tensor)
    end_time = time.time()  # 记录结束时间
    inference_time = end_time - start_time  # 计算推理时间
    print(f'\nSingle inference time for sample prompt: {inference_time:.4f} seconds')

    print(f'\nRaw model output:\n{out}')

    # 取最后一个 token 的输出 logits，计算 softmax 概率
    out_last = out[0, -1]
    probs = F.softmax(out_last.float(), dim=-1)
    print(f'\nPrompt: {prompt}')
    _, indices = torch.topk(probs, 10)
    for i in range(len(indices)):
        token_id = indices[i].item()
        token = tokenizer.decode([token_id])
        token_prob = probs[token_id].item()
        print(token, f'[probability {token_prob:.2%}]')

    ############################################################################
    # Lambada 测试
    ############################################################################
    with open("misc/lambada_test.jsonl", "r", encoding="utf-8") as f:
        # 每行一个 json 对象，分成前缀和目标 token
        todo = [json.loads(line) for line in f]
        todo = [[doc['text'].rsplit(' ', 1)[0], " " + doc['text'].rsplit(' ', 1)[1]] for doc in todo]

    print('\nCheck LAMBADA...')
    xsum = 0
    xcnt = 0
    xacc = 0
    for d in todo:
        src = [0] + tokenizer.encode(d[0]).ids
        dst = tokenizer.encode(d[1]).ids
        logits = 0
        correct = True
        # 拼接输入（前缀 + 目标），构造 tensor 时送到 NPU
        combined = src + dst
        input_seq = torch.tensor(combined, device=device).reshape(1, -1)

        start_time_1 = time.time()  # 记录开始时间
        out = model(input_seq)
        end_time_1 = time.time()  # 记录结束时间
        inference_time_1 = end_time_1 - start_time_1  # 计算推理时间
        num_tokens = input_seq.shape[1]  # 计算输入 token 数量
        time_per_token = inference_time_1 / num_tokens  # 计算每个 token 花费的时间
        print(f"Time per token: {time_per_token:.6f} seconds")

        #print(f'\nSingle inference time for sample prompt 2: {inference_time_1:.4f} seconds')


        for i in range(len(dst)):
            ooo = out[0, len(src)-1+i].float()
            prob = F.softmax(ooo, dim=-1)
            logits += math.log(prob[dst[i]])
            if torch.argmax(prob).item() != dst[i]:
                correct = False

        xcnt += 1
        xsum += logits
        xacc += 1 if correct else 0
        if xcnt % 100 == 0 or xcnt == len(todo):
            ppl = math.exp(-xsum / xcnt)
            acc = (xacc / xcnt * 100)
            print(xcnt, 'ppl', round(ppl, 2), 'acc', round(acc, 2))

################################################################################
# 释放 NPU 资源
################################################################################
acl.rt.reset_device(6)
acl.finalize()
