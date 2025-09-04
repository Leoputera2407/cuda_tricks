"""
Gpt-oss sink with sliding window with flex-attention  
that boosted Rohan Pandey's TorchTitan PR (https://github.com/pytorch/torchtitan/pull/1559/files#diff-b929013715d34f97c2006601b27c8d07f8fe4726f8cd9c3f3a5aef926b17dd5f)
from 9% to 21% MFU
"""

import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def flex_sink_attention(q, k, v, sink_size: int, window: int, *, scale=None, enable_gqa=False):
    """
    q: (B, Hq, L, D)
    k: (B, Hkv, S, D)
    v: (B, Hkv, S, Dv)
    sink_size: number of earliest tokens that are always attendable (the 'sinks')
    window: sliding window length (recent tokens, excluding sink region)
    Returns: out (B, Hq, L, Dv)
    """
    device = q.device
    B, Hq, L, _ = q.shape
    S = k.shape[2]

    # --- sink-only mask: kv < sink_size AND causal (kv <= q)
    def sink_mask(b, h, q_idx, kv_idx):
        return (kv_idx < sink_size) & (kv_idx <= q_idx)

    # --- sliding-window mask: max(0, q - window + 1) <= kv <= q, but exclude sink region to avoid double count
    # Thanks to Unsloth for figuring out the off by 1 bug (since gpt-oss window attention includes its current self)
    # https://unsloth.ai/blog/gpt-oss-context
    def win_mask(b, h, q_idx, kv_idx):
        left = torch.maximum(q_idx - (window - 1), torch.tensor(0, device=device, dtype=q_idx.dtype))
        return (kv_idx >= left) & (kv_idx <= q_idx) & (kv_idx >= sink_size)

    bm_sink = create_block_mask(sink_mask, B, Hq, L, S, device=device)           # uses block sparsity
    bm_win  = create_block_mask(win_mask,  B, Hq, L, S, device=device)

    out_sink, lse_sink = flex_attention(q, k, v, block_mask=bm_sink,
                                        scale=scale, enable_gqa=enable_gqa, return_lse=True)
    out_win,  lse_win  = flex_attention(q, k, v, block_mask=bm_win,
                                        scale=scale, enable_gqa=enable_gqa, return_lse=True)

    # logsumexp merge: stable weights per (B,H,L)
    m = torch.maximum(lse_sink, lse_win)                 # (B,H,L)
    w_sink = torch.exp(lse_sink - m)                     # proportional to denom_sink
    w_win  = torch.exp(lse_win  - m)                     # proportional to denom_win
    out = (w_sink[..., None] * out_sink + w_win[..., None] * out_win) / (w_sink + w_win)[..., None]
    return out
