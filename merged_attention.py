import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


def make_v_col_major(v: torch.Tensor) -> torch.Tensor:
    """
    FlexAttention FP8 path prefers V in 'column-major' for the last 2 dims.
    This keeps shape (B,H,S,D) but sets strides so the kernel sees col-major.
    """
    return v.transpose(-1, -2).contiguous().transpose(-1, -2)

def band_masks(S: int, window: int):
    """
    Build complementary causal masks:
      near:  0 <= (q - k) <= window
      far:   (q - k) >  window
    """
    def near_fn(b, h, q_idx, k_idx):
        diff = q_idx - k_idx
        return (diff >= 0) & (diff <= window)

    def far_fn(b, h, q_idx, k_idx):
        diff = q_idx - k_idx
        return (diff >= 0) & (diff >  window)

    near = create_block_mask(near_fn, B=None, H=None, Q_LEN=S, KV_LEN=S)
    far  = create_block_mask(far_fn , B=None, H=None, Q_LEN=S, KV_LEN=S)
    return near, far

def lse_merge(z1, lse1, z2, lse2, out_dtype):
    """
    Exact merge of two masked attentions using per-row log-sum-exp (LSE).

    # Inspired from https://flashinfer.ai/2024/02/02/cascade-inference.html
    # lse of attention is communicative and distributive, and maps to full attention
    # Better numerical stability too if you minus the max.
    Shapes:
      z*   : [B, S, H, D]
      lse* : [B, H, S]  (log(sum_j exp(logits_ij)))
    """
    m  = torch.maximum(lse1, lse2)                 # [B,H,S]
    w1 = torch.exp(lse1 - m)                       # [B,H,S]
    w2 = torch.exp(lse2 - m)
    num = z1 * w1[..., None] + z2 * w2[..., None]  # [B,S,H,D]
    den = (w1 + w2)[..., None]
    return (num / den).to(out_dtype)


@torch.no_grad()
def merged_flex_attn_bf16_near_fp8_far(
    q_bf16: torch.Tensor, 
    k_bf16: torch.Tensor, 
    v_bf16: torch.Tensor,
    window: int,
    return_lse: bool = False
):
    """
    Runs FlexAttention twice with complementary masks, then LSE-merges results:
      - NEAR  (|q-k| <= window, causal) in bf16
      - FAR   (q>k and (q-k) > window) with FP8 K/V
    Q stays in bf16 for stability. Output matches global softmax(QK)V up to FP8 error.
    """
    assert q_bf16.dtype == torch.bfloat16
    B, S, H, D = q_bf16.shape
    scale = D ** -0.5

    near_mask, far_mask = band_masks(S, window)

    # NEAR pass (bf16 all)
    near_out, near_lse = flex_attention(
        q_bf16, k_bf16, v_bf16,
        block_mask=near_mask, scale=scale,
        return_lse=True
    )  # near_out: [B,S,H,D], near_lse: [B,H,S]

    # FAR pass (FP8 K/V). Keep Q in bf16.
    # https://arxiv.org/pdf/2506.08027 (best perf use e4m3)
    k_fp8 = k_bf16.contiguous().to(torch.float8_e4m3fn)
    v_fp8 = make_v_col_major(v_bf16).to(torch.float8_e4m3fn)

    far_out, far_lse = flex_attention(
        q_bf16, k_fp8, v_fp8,
        block_mask=far_mask, scale=scale,
        return_lse=True
    )
    # For merge-stability
    far_out = far_out.to(torch.bfloat16)

    out = lse_merge(near_out, near_lse, far_out, far_lse, out_dtype=q_bf16.dtype)

    if return_lse:
        # Global LSE for diagnostics (log(sum exp over near∪far)):
        # log( exp(lse1) + exp(lse2) ) computed in the common log-base 'm'
        m  = torch.maximum(near_lse, far_lse)
        lse_global = m + torch.log(torch.exp(near_lse - m) + torch.exp(far_lse - m))
        return out, lse_global
    return out

if __name__ == "__main__":
    torch.set_default_device("cuda")
    B, S, H, D = 2, 2048, 16, 64
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    W = 256

    ref = flex_attention(q, k, v, return_lse=False)

    out = merged_flex_attn_bf16_near_fp8_far(q, k, v, window=W)

    print("max |diff| vs bf16 full:", (ref - out).abs().max().item())
