import torch

torch.manual_seed(0)

B, T, D = 2, 3, 8
H = 2
Dh = D // H

x = torch.randn(B, T, D)

# 除以 sqrt(D)，让这个小实验的初始数值尺度较温和
wq = torch.randn(D, D) / (D ** 0.5)
wk = torch.randn(D, D) / (D ** 0.5)
wv = torch.randn(D, D) / (D ** 0.5)

# A：生成 Q、K、V
q = x @ wq
k = x @ wk
v = x @ wv

print("投影后的 q:", q.shape)

# B：分别从 [B, T, D] 变成 [B, H, T, Dh]
q = q.reshape(B, T, H, Dh).transpose(2,1)
k = k.reshape(B, T, H, Dh).transpose(2,1)
v = v.reshape(B, T, H, Dh).transpose(2,1)

print("拆头后的 q:", q.shape)

# C：计算匹配分数、缩放、softmax
scores = q @ k.transpose(-2, -1)
weights = torch.softmax(scores.divide(Dh ** 0.5),dim=-1)

print("scores:", scores.shape)
print("每个 query 的权重之和:", weights.sum(dim=-1))

# D：汇总 V，再合并各个头
head_output = weights @ v
output = head_output.transpose(2,1).reshape(B, T, D)

print("head_output:", head_output.shape)
print("output:", output.shape)

# 单独核对：第 0 条文本、第 0 个头、第 0 个 query 的输出
manual = (
    weights[0, 0, 0, 0] * v[0, 0, 0]
    + weights[0, 0, 0, 1] * v[0, 0, 1]
    + weights[0, 0, 0, 2] * v[0, 0, 2]
)

print(
    "加权求和是否一致:",
    torch.allclose(head_output[0, 0, 0], manual, atol=1e-6),
)