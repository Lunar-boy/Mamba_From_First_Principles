import torch
from einops import einsum

def ssd_naive(X, A, B, C, initial_state=None):
    """
    X: (batch, seqlen, nheads, headdim)
    A: (batch, seqlen, nheads)  
    -- 注意：Mamba2 中 A 是标量衰减因子 (经 discretize 后)
    B: (batch, seqlen, nheads, d_state)
    C: (batch, seqlen, nheads, d_state)
    """
    batch, seqlen, nheads, headdim = X.shape
    d_state = B.shape[-1]

    # 状态 h 的形状: (batch, nheads, headdim, d_state)
    h = torch.zeros(batch, nheads, headdim, d_state, dtype=X.dtype, 
                    device=X.device)
    if initial_states is not None:
        h = h + initial_states

    outputs = []    

    for t in range(seqlen):
        # 离散化后的衰减因子 exp(A_t*dt) 传入前已处理为log-space 或者
        # 直接为衰减系数 a_t
        a_t = torch.exp(A[:,t]) # (batch, nheads)

        # 状态更新 h_t = a_t * h_{t-1} + x_t \otimes B_t 外积
        BX = einsum(B[:, t], X[:, t],"b h n, b h p -> b h p n")
        h = einsum(a_t, h, "b h, b h p n -> b h p") + BX

        # 输出投影：y_t = C_t h_t
        y_t = einsum(C[:, t], h, "b h n, b h p n -> b h p")
        outputs.append(y_t)

    Y = torch.stack(outputs, dim=1) # (batch, seqlen, nheads, headdim)
    return Y, h