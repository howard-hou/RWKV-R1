import torch
import time

# 设定超参数
HEAD_SIZE = 64
B = 2  # Batch size
T = 2  # Sequence length
C = 128  # Number of channels, for simplicity in this example

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(22)
# 生成测试数据
r = torch.empty(B, T, C).uniform_(-1, 1).cuda()  # (Batch, Time, Channels)
w = torch.empty(B, T, C).uniform_(-8, -6).cuda()  # (Batch, Time, Channels)
k = torch.empty(B, T, C).uniform_(-1, 1).cuda()  # (Batch, Time, Channels)
v = torch.empty(B, T, C).uniform_(-1, 1).cuda()  # (Batch, Time, Channels)
a = torch.empty(B, T, C).uniform_(-1, 1).cuda()  # (Batch, Time, Channels)
b = torch.empty(B, T, C).uniform_(-1, 1).cuda()  # (Batch, Time, Channels)

# 状态初始化
H = C // HEAD_SIZE
N = HEAD_SIZE
state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)
state_cuda = state.clone() # copy the state


# RWKV7_OP pytorch 实现 ########################################################
def RWKV7_OP(r, w, k, v, a, b, state):
    B, T, C = r.size()
    H = C // HEAD_SIZE
    N = HEAD_SIZE
    r = r.view(B, T, H, N).float()
    k = k.view(B, T, H, N).float()
    v = v.view(B, T, H, N).float()
    a = a.view(B, T, H, N).float()
    b = b.view(B, T, H, N).float()
    w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
    out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)

    for t in range(T):
        kk = k[:, t, :].view(B, H, 1, N)
        rr = r[:, t, :].view(B, H, N, 1)
        vv = v[:, t, :].view(B, H, N, 1)
        aa = a[:, t, :].view(B, H, N, 1)
        bb = b[:, t, :].view(B, H, 1, N)
        state = state * w[:, t, :, None, :] + state @ aa @ bb + vv @ kk
        out[:, t, :] = (state @ rr).view(B, H, N)

    return out.view(B, T, C), state

# 调用 RWKV7_OP 并获取结果
start = time.time()
out_naive, state_naive = RWKV7_OP(r, w, k, v, a, b, state)
end = time.time()
print("OP time:", end - start)

# 打印输出形状
print("Output shape:", out_naive.shape)  # 应该是 (B, T, C)
print("Final state shape:", state_naive.shape)  # 应该是 (B, H, N, N)


# CUDA 实现 ####################################################################
DTYPE = torch.half # better
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

def RWKV7_CUDA(r, w, k, v, a, b, s):
    return WKV_7.apply(r, w, k, v, a, b, s)

#  to half precision
r = r.half()
w = w.half()
k = k.half()
v = v.half()
a = a.half()
b = b.half()
state_cuda = state_cuda.half()

# 调用 RWKV7_OP 并获取结果
start = time.time()
out_cuda, state_cuda = RWKV7_CUDA(r, w, k, v, a, b, state_cuda)
end = time.time()
print("CUDA time:", end - start)

print("Sequnce length:", T)
# calculate the error
for t in range(T-1, T):
    print("Step:", t)
    print("y Max Error:", (out_cuda[:, t, :] - out_naive[:, t, :]).abs().max())
    print("y Mean Error:", (out_cuda[:, t, :] - out_naive[:, t, :]).abs().mean())

print("state Max Error:", (state_cuda - state_naive).abs().max())
print("state Mean Error:", (state_cuda - state_naive).abs().mean())