########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import torch, types, os, time, math, json
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
np.set_printoptions(precision=4, suppress=True, linewidth=200)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
# torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch._C._jit_set_autocast_mode(False)

'''
This will load RWKV-7 "Goose" x070 and inference in GPT-mode (slower than RNN-mode for autoregressive generation)
'''

args = types.SimpleNamespace()

# model download: https://huggingface.co/BlinkDL/rwkv-7-world

MODEL_PATH = "/root/RWKV-LM/RWKV-v7/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth"

# for 0.1B
args.n_layer = 12
args.n_embd = 768
D_DECAY_LORA = 64
D_AAA_LORA = 64
D_MV_LORA = 32
D_GATE_LORA = 128

args.vocab_size = 65536

# DTYPE = torch.bfloat16
DTYPE = torch.half # better

args.head_size_a = 64 # don't change
HEAD_SIZE = args.head_size_a

USE_CUDA_KERNEL = True # False => UNOPTIMIZED, VERY SLOW

def __nop(ob):
    return ob

MyModule = nn.Module
MyFunction = __nop


########################################################################################################
# RWKV State Management
########################################################################################################    
class TimeMixState:
    def __init__(self, shift_state: torch.Tensor, wkv_state: torch.Tensor):
        self.shift_state = shift_state
        self.wkv_state = wkv_state


class ChannelMixState:
    def __init__(self, shift_state: torch.Tensor):
        self.shift_state = shift_state


class BlockState:
    def __init__(self, time_mix_state: TimeMixState,
                 channel_mix_state: ChannelMixState):
        self.time_mix_state = time_mix_state
        self.channel_mix_state = channel_mix_state

class BlockStateList:

    def __init__(self, shift_states, wkv_states):
        self.wkv_states = wkv_states
        self.shift_states = shift_states

    @staticmethod
    def empty(N, B, C, H, device, dtype):
        wkv_states = torch.zeros((N, B, H, C//H, C//H),
                                 device=device,
                                 dtype=dtype)
        shift_states = torch.zeros((N, 2, B, 1, C), device=device, dtype=dtype)
        return BlockStateList(shift_states, wkv_states)

    def __getitem__(self, layer: int):
        return BlockState(
            TimeMixState(self.shift_states[layer, 0], self.wkv_states[layer]),
            ChannelMixState(self.shift_states[layer, 1]))

    def __setitem__(self, layer: int, state: BlockState):
        self.shift_states[layer, 0] = state.time_mix_state.shift_state
        self.wkv_states[layer] = state.time_mix_state.wkv_state
        self.shift_states[layer, 1] = state.channel_mix_state.shift_state


########################################################################################################
# RWKV Tokenizer (slow version)
########################################################################################################

class RWKV_TOKENIZER():
    table: list[list[list[bytes]]]
    good: list[set[int]]
    wlen: list[int]
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = [] # must be already sorted
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        # precompute some tables for fast matching
        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]

        for i in reversed(range(len(sorted))): # reverse order - match longer tokens first
            s = sorted[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes) -> list[int]:
        src_len: int = len(src)
        tokens: list[int] = []
        i: int = 0
        while i < src_len:
            s: bytes = src[i : i + 1]

            if i < src_len - 1:
                s1: int = int(src[i + 1])
                s0: int = int(src[i])
                if s1 in self.good[s0]:
                    sss: bytes = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)

        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
            # print(repr(s), i)
        print()

tokenizer = RWKV_TOKENIZER("rwkv_vocab_v20230424.txt")

########################################################################################################
# CUDA Kernel
########################################################################################################

if USE_CUDA_KERNEL:

    from torch.utils.cpp_extension import load

    load(name="wkv7", sources=["cuda/wkv7state_op.cpp", f"cuda/wkv7state_cuda.cu"], is_python_module=False,
                        verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
    class WKV_7(torch.autograd.Function):
        @staticmethod
        def forward(ctx, r, w, k, v, a, b, s):
            with torch.no_grad():
                B, T, C = r.size()
                H = C // HEAD_SIZE
                N = HEAD_SIZE
                assert HEAD_SIZE == C // H
                assert r.dtype == DTYPE
                assert w.dtype == DTYPE
                assert k.dtype == DTYPE
                assert v.dtype == DTYPE
                assert a.dtype == DTYPE
                assert b.dtype == DTYPE
                assert s.dtype == DTYPE
                assert r.is_contiguous()
                assert w.is_contiguous()
                assert k.is_contiguous()
                assert v.is_contiguous()
                assert a.is_contiguous()
                assert b.is_contiguous()
                assert s.is_contiguous()
                y = torch.empty((B, T, C), device=k.device, dtype=DTYPE, memory_format=torch.contiguous_format)
                torch.ops.wkv7.forward(B, T, C, H, r, w, k, v, a, b, y, s)
                return y, s

    def RWKV7_OP(r, w, k, v, a, b, s):
        return WKV_7.apply(r, w, k, v, a, b, s)

else:

    def RWKV7_OP(r, w, k, v, a, b, state):
        B, T, C = r.size()
        H = C // HEAD_SIZE
        N = HEAD_SIZE
        r = r.view(B, T, H, N).float()
        k = k.view(B, T, H, N).float()
        v = v.view(B, T, H, N).float()
        a = a.view(B, T, H, N).float()
        b = b.view(B, T, H, N).float()
        state = state.float()
        w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
        out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)

        for t in range(T):
            kk = k[:, t, :].view(B, H, 1, N)
            rr = r[:, t, :].view(B, H, N, 1)
            vv = v[:, t, :].view(B, H, N, 1)
            aa = a[:, t, :].view(B, H, N, 1)
            bb = b[:, t, :].view(B, H, 1, N)
            state = state * w[: , t, :, None, :] + state @ aa @ bb + vv @ kk
            out[:, t, :] = (state @ rr).view(B, H, N)

        return out.view(B, T, C).to(dtype=DTYPE), state.to(dtype=DTYPE)

########################################################################################################
# RWKV TimeMix
########################################################################################################

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

        self.x_r = nn.Parameter(torch.empty(1,1,C))
        self.x_w = nn.Parameter(torch.empty(1,1,C))
        self.x_k = nn.Parameter(torch.empty(1,1,C))
        self.x_v = nn.Parameter(torch.empty(1,1,C))
        self.x_a = nn.Parameter(torch.empty(1,1,C))
        self.x_g = nn.Parameter(torch.empty(1,1,C))

        self.w0 = nn.Parameter(torch.empty(1,1,C))
        self.w1 = nn.Parameter(torch.empty(C, D_DECAY_LORA))
        self.w2 = nn.Parameter(torch.empty(D_DECAY_LORA, C))

        self.a0 = nn.Parameter(torch.empty(1,1,C))
        self.a1 = nn.Parameter(torch.empty(C, D_AAA_LORA))
        self.a2 = nn.Parameter(torch.empty(D_AAA_LORA, C))

        self.v0 = nn.Parameter(torch.empty(1,1,C))
        self.v1 = nn.Parameter(torch.empty(C, D_MV_LORA))
        self.v2 = nn.Parameter(torch.empty(D_MV_LORA, C))

        self.g1 = nn.Parameter(torch.empty(C, D_GATE_LORA))
        self.g2 = nn.Parameter(torch.empty(D_GATE_LORA, C))

        self.k_k = nn.Parameter(torch.empty(1,1,C))
        self.k_a = nn.Parameter(torch.empty(1,1,C))
        self.r_k = nn.Parameter(torch.empty(H,N))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!


    def token_shift(self, x, v_first, shift_state):
        B, T, C = x.size()
        xx = torch.concat((shift_state, x[:, :-1]), dim=1) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
           v_first = v # store the v of the first layer
        else:
           v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)
        return r, w, k, v, kk, a, g, v_first, x[:, -1:]

    @MyFunction
    def forward(self, x, v_first, last_state: TimeMixState):
        B, T, C = x.size()
        H = self.n_head

        r, w, k, v, kk, a, g, v_first, lx = self.token_shift(x, v_first, last_state.shift_state)

        wkv_state = last_state.wkv_state.clone().contiguous()
        x, wkv_state = RWKV7_OP(r, w, k, v, -kk, kk*a, wkv_state)
        # print(self.layer_id, x[0, -1].sum())
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)
        
        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first, TimeMixState(lx, wkv_state)
    
########################################################################################################
# RWKV ChannelMix
########################################################################################################

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
    def forward(self, x, last_state: ChannelMixState):
        xx = torch.concat((last_state.shift_state, x[:, :-1]), dim=1) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k), ChannelMixState(x[:, -1:])

########################################################################################################
# RWKV Block
########################################################################################################

class Block(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln0 = nn.LayerNorm(args.n_embd) # only used in block 0, should be fused with emb
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)
        
    @MyFunction
    def forward(self, x, v_first, last_state: BlockState):

        if self.layer_id == 0:
            x = self.ln0(x)

        x_attn, v_first, att_state = self.att(self.ln1(x), v_first, last_state.time_mix_state)
        x = x + x_attn

        ffn_out ,ffn_state = self.ffn(self.ln2(x), last_state.channel_mix_state)
        x = x + ffn_out
        return x, v_first, BlockState(att_state, ffn_state)

########################################################################################################
# RWKV Model
########################################################################################################

class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        args.dim_att = args.n_embd
        args.dim_ffn = args.n_embd * 4
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def forward(self, idx, states=None):

        x = self.emb(idx)

        if states is None:
            B, T, C = x.size()
            H = C // HEAD_SIZE
            N = HEAD_SIZE
            n_layer = len(self.blocks)
            states = BlockStateList.empty(n_layer, B, C, H, device=x.device, dtype=x.dtype)

        v_first = torch.zeros_like(x)
        for i, block in enumerate(self.blocks):
            x, v_first, s_new = block(x, v_first, states[i])
            states[i] = s_new

        x = self.ln_out(x)
        x = self.head(x)

        return x, states

########################################################################################################
# RWKV Inference
########################################################################################################

model_params = torch.load(MODEL_PATH, map_location="cpu")

with torch.no_grad():

    model = RWKV(args).to(dtype=DTYPE).cuda()
    model.load_state_dict(model_params, strict=False) # we will ignore blocks.0.att.v0/v1/v2
    ########################################################################################################
    # init states
    B = 1
    C = args.n_embd
    H = C // HEAD_SIZE
    N = HEAD_SIZE
    states = BlockStateList.empty(args.n_layer, B, C, H, device="cuda", dtype=DTYPE)
    ########################################################################################################

    prompt = "The Eiffel tower is in the city of"
    input = tokenizer.encode(prompt)
    print(f'\nInput:\n{input}')

    _, prefill_states = model.forward(torch.tensor(input[:-1]).reshape(1,-1).cuda())
    print(prefill_states[0].time_mix_state.wkv_state.sum())
    exit()
    out, _ = model.forward(torch.tensor(input[-1:]).reshape(1,-1).cuda(), prefill_states)
    print(f'\nOutput:\n{out}')

    # logits of the last token => prediction for the next token    
    out = out[0, -1]
    
    probs = F.softmax(out.float(), dim=-1) # compute softmax in float (more accurate)

    print(f'\n{prompt}')

    _, indices = torch.topk(probs, 10) # print top-10 possibilities
    for i in range(len(indices)):
        token_id = indices[i].item()
        token = tokenizer.decode([token_id])
        token_prob = probs[token_id].item()
        print(token, f'[probability {token_prob:.2%}]')

    ########################################################################################################

    with open(f"misc/lambada_test.jsonl", "r", encoding="utf-8") as f:
        todo = [json.loads(line) for line in f]
        todo = [[doc['text'].rsplit(' ', 1)[0], " " + doc['text'].rsplit(' ', 1)[1]] for doc in todo]

    print('\nCheck LAMBADA...')
    xsum = 0
    xcnt = 0
    xacc = 0
    xstart = time.time()
    for d in todo:
        src = [0] + tokenizer.encode(d[0])
        dst = tokenizer.encode(d[1])

        logits = 0
        correct = True
        _, prefill_states = model.forward(torch.tensor(src[:-1]).reshape(1,-1).cuda())
        out, _ = model.forward(torch.tensor(src[-1:]+dst).reshape(1,-1).cuda(), prefill_states)
        for i in range(len(dst)):
            # logits of the last token => prediction for the next token
            ooo = out[0,i].float() # the output of last src
            probs = F.softmax(ooo, dim=-1)
            logits += math.log(probs[dst[i]])
            if torch.argmax(probs).item() != dst[i]:
                correct = False

        xcnt += 1
        xsum += logits
        xacc += 1 if correct else 0
        if xcnt % 100 == 0 or xcnt == len(todo):
            print(xcnt, 'ppl', round(math.exp(-xsum / xcnt), 2), 'acc', round(xacc/xcnt*100, 2))
            print(round((time.time()-xstart)/xcnt, 3), 'sec/iter')
