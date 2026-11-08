"""
FNR-100M: Full model implementation with all optimizations.
Optimizations over the original paper:
  1. Trainable Value projection (was identity)
  2. k=4 (was k=2): denser routing signal
  3. RoPE positional encoding
  4. RMSNorm (was LayerNorm)
  5. GELU operator activation (was Sigmoid)
  6. Grouped GEMM: batched top-k operator apply
  7. Progressive LoRA: r=0->1->2 to escape ceiling gradually
  8. Adaptive entropy: collapse detection with soft penalty
  9. Tied embedding/head
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass
from config import FNRConfig
from theory_modules import (ResidualEscape, LibraryGrowthMonitor,
                            spectral_coverage_init, lora_tube_radius)
from ctrl_modules import FunctionCalculator, PathController, deterministic_orthonormal


# ─────────────────────────────────────────────
#  RMSNorm
# ─────────────────────────────────────────────
class RMSNorm(nn.Module):
    """RMSNorm。`freeze=True`（universal 固定程序）时 weight 注册为全 1 的
    不可训练 buffer —— 删掉 L 份逐层重复的单位缩放可训练向量，只保留
    纯固定的归一化计算（归一化本身是"程序"，不是变量）。"""
    def __init__(self, dim: int, eps: float = 1e-5, freeze: bool = False):
        super().__init__()
        self.eps = eps
        if freeze:
            self.register_buffer("weight", torch.ones(dim))
            self._frozen_norm = True
        else:
            self.weight = nn.Parameter(torch.ones(dim))
            self._frozen_norm = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() / rms).type_as(x) * self.weight


# ─────────────────────────────────────────────
#  RoPE
# ─────────────────────────────────────────────
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0, max_seq: int = 1024):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0):
        """q, k: (B, N, D)  -- no head dim; RoPE applied directly."""
        N = q.size(1)
        cos = self.cos[start_pos:start_pos + N].to(q.dtype)  # (N, D)
        sin = self.sin[start_pos:start_pos + N].to(q.dtype)
        D = q.size(-1)

        def rotate(x):
            x1, x2 = x[..., :D // 2], x[..., D // 2:]
            return torch.cat([-x2, x1], dim=-1)

        q = q * cos + rotate(q) * sin
        k = k * cos + rotate(k) * sin
        return q, k


# ─────────────────────────────────────────────
#  Frozen Operator Bank + Progressive LoRA
# ─────────────────────────────────────────────
class FrozenOperatorBank(nn.Module):
    """
    K frozen operators, f_v(x) = act(W_v x + b_v).
    LoRA (Optimization #7): W_v_eff = W_v + lora_scale * B_v @ A_v
    A_v: (K, r, d), B_v: (K, d, r). Zero-init so effective = W_v at start.
    Progressive: lora_scale is a function of training step (0 -> 1 -> 2).
    """
    def __init__(self, K: int, d: int, lora_r: int = 0,
                 activation: str = "gelu", lora_alpha: float = 1.0,
                 frozen_fp16: bool = False, mb_budget: int = 96,
                 rank: int = 0):
        super().__init__()
        # Frozen weights: registered as buffers (not parameters)
        self.register_buffer("W", torch.empty(K, d, d))
        self.register_buffer("b", torch.zeros(K, d))
        self._orthogonal_init()
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

        if lora_r > 0:
            # A: (K, r, d)  B: (K, d, r)
            self.lora_A = nn.Parameter(torch.zeros(K, lora_r, d))
            self.lora_B = nn.Parameter(torch.zeros(K, d, lora_r))
            with torch.no_grad():
                self.lora_A.normal_(0, 1e-3)

        self._activation = activation
        self.frozen_fp16 = frozen_fp16          # run the frozen GEMM in fp16
        self.mb_budget = mb_budget              # VRAM budget (MB) for live W_sel
        # 低秩因子化（0 可训练变量）：rank>0 时调用 factorize() 把 W(d×d) 换成
        # Wa(d×r)/Wb(r×d) 两个小因子，运行期 gather 字节与 MAC 同降。
        self.rank = int(rank)
        self._Wfull_cache = None

    def _orthogonal_init(self):
        K, d, _ = self.W.shape
        with torch.no_grad():
            for i in range(K):
                self.W[i] = torch.linalg.qr(
                    torch.randn(d, d, device=self.W.device)
                )[0].contiguous()

    @property
    def d(self) -> int:
        return self.W.shape[-1]

    @property
    def K(self) -> int:
        return self.W.shape[0]

    def effective_W(self, lora_scale: float = 1.0) -> torch.Tensor:
        """W + scale * B @ A. lora_scale=0 gives pure frozen.
        B_v: (K, d, r)  A_v: (K, r, d)  →  B@A: (K, d, d).
        低秩模式下 base = W_full()（从因子重建的全空间 K×d×d）。"""
        base = self.W_full()
        if self.lora_r > 0 and lora_scale > 0.0:
            delta = torch.einsum("kjr,krd->kjd", self.lora_B, self.lora_A)
            return base + (self.lora_alpha / self.lora_r) * lora_scale * delta
        return base

    def W_full(self) -> torch.Tensor:
        """全空间 K×d×d：full 模式直接返回 buffer；低秩模式从因子重建（缓存）。
        W_v = Wa_v (d×r) @ Wb_v (r×d) → (d×d)。"""
        if getattr(self, "rank", 0) > 0:
            if self._Wfull_cache is None:
                self._Wfull_cache = torch.einsum("var,vrb->vab",
                                                  self.Wa, self.Wb)
            return self._Wfull_cache
        return self.W

    def factorize(self) -> int:
        """把全空间 bank（W 已含谱/正交初始化结果）SVD 截断成
        Wa(K,d,r)·Wb(K,r,d)。0 可训练变量、离线一次性。返回实际秩。"""
        if getattr(self, "rank", 0) <= 0:
            return 0
        K, d, _ = self.W.shape
        r = min(self.rank, d)
        with torch.no_grad():
            Wa = torch.empty(K, d, r, device=self.W.device, dtype=self.W.dtype)
            Wb = torch.empty(K, r, d, device=self.W.device, dtype=self.W.dtype)
            for v in range(K):
                U, S, Vh = torch.linalg.svd(self.W[v], full_matrices=False)
                sq = torch.sqrt(S[:r])
                Wa[v] = U[:, :r] * sq
                Wb[v] = sq.unsqueeze(1) * Vh[:r]
            del self._buffers["W"]
            self.register_buffer("Wa", Wa.contiguous())
            self.register_buffer("Wb", Wb.contiguous())
        self._Wfull_cache = None
        self._cache = {}
        return r

    def frozen_numel(self) -> int:
        """参数账：低秩模式只数因子（不再是 K·d²）。"""
        if getattr(self, "rank", 0) > 0:
            return self.Wa.numel() + self.Wb.numel() + self.b.numel()
        return self.W.numel() + self.b.numel()

    def _cast_factors(self, ref: torch.Tensor):
        """低秩因子 dtype 缓存（等价 full 模式的 _cast_pair）。"""
        dt = ref.dtype
        if self.Wa.dtype == dt:
            return self.Wa, self.Wb, self.b.to(dt)
        fcache = getattr(self, "_fcache", {})
        if dt not in fcache:
            fcache[dt] = (self.Wa.to(dt), self.Wb.to(dt), self.b.to(dt))
            self._fcache = fcache
        return fcache[dt]

    def _cast_pair(self, ref: torch.Tensor):
        """Return (W, b) in ref.dtype, CACHED per dtype so the 93M-param
        float32->fp16 conversion happens ONCE, not every forward step.
        This is the main throughput fix on MX-class GPUs."""
        dt = ref.dtype
        if self.W.dtype == dt:
            return self.W, self.b
        cache = getattr(self, "_cache", {})
        if dt not in cache:
            cache[dt] = (self.W.to(dt), self.b.to(dt))
            self._cache = cache
        return cache[dt]

    def forward_grouped(self, x: torch.Tensor, indices: torch.Tensor,
                        weights: torch.Tensor, lora_scale: float = 1.0,
                        gemm_chunk: int = 0) -> torch.Tensor:
        """
        Grouped GEMM (Optimization #6): batched top-k operator apply.
        Chunked over tokens to bound GPU memory: W_sel is (chunk,k,D,D).
        x:       (B, N, D)
        indices: (B, N, k)
        weights: (B, N, k)
        """
        B, N, D = x.shape
        k = indices.shape[-1]
        b = B * N
        x_flat = x.reshape(b, D)

        # ── 低秩因子路径（bank_rank>0 且 lora_r==0，0 可训练变量）：
        #    gather 2·k·d·r 字节/token（原 k·d²），大 batched GEMM 拆成两个
        #    宽 M 小 GEMM（d×r + r×d），MAC 量降至 2·d·r/槽位（原 d²）。
        #    门槛用 lora_r==0（lora_scale 在 lora_r==0 时 progressive 恒返回 1.0）。
        if getattr(self, "rank", 0) > 0 and self.lora_r == 0:
            Wa, Wb, b_use = self._cast_factors(x_flat)
            r = Wa.shape[2]
            idx = indices.reshape(b, k)
            wts = weights.reshape(b, k, 1)
            per_tok = (2 * k * r * D + k * D) * x_flat.element_size()
            budget = self.mb_budget * 1024 * 1024
            chunk = max(32, min(b, budget // max(per_tok, 1)))
            out = torch.empty(b, D, device=x.device, dtype=x_flat.dtype)
            for s in range(0, b, chunk):
                e = min(s + chunk, b)
                Wsa = Wa[idx[s:e]]                       # (C,k,d,r)
                Wsb = Wb[idx[s:e]]                       # (C,k,r,d)
                h = torch.einsum("bd,bkdi->bki", x_flat[s:e], Wsa)
                y = torch.einsum("bki,bkid->bkd", h, Wsb) + b_use[idx[s:e]]
                y = self._act(y)
                out[s:e] = (y * wts[s:e].to(x_flat.dtype)).sum(1)
            return out.view(B, N, D)

        # Cast the frozen (W,b) ONCE per dtype (cached), not per step.
        # LoRA-active path: recompute effective W (only when lora_scale>0).
        if lora_scale > 0.0:
            W_eff = self.effective_W(lora_scale).to(x_flat.dtype)
            W_use, b_use = W_eff, self.b.to(x_flat.dtype)
        else:
            W_use, b_use = self._cast_pair(x_flat)

        # 固定程序化提速（frozen_fp16）：冻结算子库不训练，fp16 GEMM 只是
        # 确定性舍入（≈量化噪声），在 Pascal（FP16 FLOPs=2×FP32、W_sel
        # 字节减半）上加速，且不引入任何可训练变量。仅 LoRA 关闭时启用。
        if self.frozen_fp16 and lora_scale <= 0.0 and x_flat.dtype != torch.float16:
            fp16_cache = getattr(self, "_cache", {}).get(torch.float16)
            if fp16_cache is None:
                self._cache = getattr(self, "_cache", {})
                self._cache[torch.float16] = (self.W.half(), self.b.half())
                fp16_cache = self._cache[torch.float16]
            W_use, b_use = fp16_cache
            x_flat = x_flat.half()

        idx = indices.reshape(b, k)
        wts = weights.reshape(b, k, 1)

        # VRAM-adaptive chunk: bound the W_sel working set to ~96 MB so that
        # large batches (e.g. B4/S128, b=512) do not materialise a 512-token
        # gather (chunk*k*D*D*4 bytes ≈ 852 MB) that thrashes a 2 GB card.
        if gemm_chunk > 0:
            chunk = gemm_chunk
        else:
            wsel_per_tok = k * D * D * 4          # bytes for one token's k×D×D slice
            budget = self.mb_budget * 1024 * 1024  # live W_sel budget in MB
            chunk = max(32, min(b, budget // wsel_per_tok))

        out = torch.empty(b, D, device=x.device, dtype=x_flat.dtype)
        for s in range(0, b, chunk):
            e = min(s + chunk, b)
            C = e - s
            W_sel = W_use[idx[s:e]]                       # (C, k, D, D)
            b_sel = b_use[idx[s:e]]                       # (C, k, D)
            # Grouped GEMM: (C,D) x (C,k,D,D) -> (C,k,D).
            # Theoretical-limit: einsum maps to a single batched GEMM call over
            # the (C,k) groups without materialising a broadcast copy of x.
            y = torch.einsum("bd,bkij->bkj", x_flat[s:e], W_sel) + b_sel
            y = self._act(y)
            out[s:e] = (y * wts[s:e].to(x_flat.dtype)).sum(1)
        return out.view(B, N, D)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self._activation == "gelu":
            return F.gelu(x)
        elif self._activation == "relu":
            return F.relu(x)
        return torch.sigmoid(x)

    def extra_repr(self):
        return f"K={self.K}, d={self.d}, lora_r={self.lora_r}, act={self._activation}"


# ─────────────────────────────────────────────
#  Context Router (Attention + RoPE + Trainable V)
# ─────────────────────────────────────────────
class ContextRouter(nn.Module):
    """
    Paper: Q,K in d_r space, V = identity (x passes through unchanged).
    Optimization #1: V is now a trainable linear projection d -> d_v -> d.
    RoPE applied in d_r space for Q,K.
    """
    def __init__(self, cfg: FNRConfig):
        super().__init__()
        d, d_r, d_v = cfg.d_model, cfg.d_router, cfg.d_v
        self.q_proj = nn.Linear(d, d_r, bias=False)
        self.k_proj = nn.Linear(d, d_r, bias=False)
        self.v_proj = nn.Linear(d, d_v, bias=False)   # OPT #1
        self.o_proj = nn.Linear(d_v, d, bias=False)
        self.rope = RotaryEmbedding(d_r, cfg.rope_theta, cfg.max_seq_len)
        self.scale = math.sqrt(d_r)
        # Cache: avoid rebuilding the causal mask every layer/step
        self._mask_cache = None

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        B, N, _ = x.shape
        q = self.q_proj(x)   # (B, N, d_r)
        k = self.k_proj(x)   # (B, N, d_r)
        v = self.v_proj(x)   # (B, N, d_v)

        q, k = self.rope(q, k, start_pos)

        attn = torch.bmm(q, k.transpose(1, 2)) / self.scale
        # cached causal mask (True = keep); only rebuilt when N or device changes
        mkey = (N, attn.device)
        if getattr(self, "_mask_key", None) != mkey:
            self._mask_cache = (torch.tril(
                torch.ones(N, N, device=attn.device, dtype=torch.bool))
                .unsqueeze(0)
            )
            self._mask_key = mkey
        attn = attn.masked_fill(~self._mask_cache, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v)
        return self.o_proj(out)

    def forward_cached(self, x: torch.Tensor, start_pos: int,
                       cache: dict) -> torch.Tensor:
        """
        Incremental attention with a persistent KV cache (for fast autoregressive
        generation). x: (B, N, d) with N new tokens; cache stores running K,V.
        Returns: (B, N, d) identical to self.forward over the full prefix.
        """
        B, N, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q, k = self.rope(q, k, start_pos)

        # append new K,V
        if "k" in cache:
            cache["k"] = torch.cat([cache["k"], k], dim=1)
            cache["v"] = torch.cat([cache["v"], v], dim=1)
        else:
            cache["k"], cache["v"] = k, v

        K_all = cache["k"]           # (B, L, d_r)
        V_all = cache["v"]           # (B, L, d_v)
        L = K_all.size(1)
        attn = torch.bmm(q, K_all.transpose(1, 2)) / self.scale

        # causal mask over (N_new, L_total): query i (global pos start_pos+i)
        # may attend to K positions j <= start_pos + i.
        j = torch.arange(L, device=attn.device)                       # (L,)
        i = start_pos + torch.arange(N, device=attn.device)          # (N,)
        allow = j[None, :] <= i[:, None]                              # (N, L) bool
        attn = attn.masked_fill(~allow[None], float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, V_all)
        return self.o_proj(out)

    # graph-safe precomputed structures (for CUDA-Graph decode)
    def _init_static(self, L_fixed: int, device):
        self._L_fixed = L_fixed
        # causal mask row g: query at global pos g may attend keys j <= g
        self.register_buffer("_mask_full",
                             torch.tril(torch.ones(L_fixed, L_fixed, device=device,
                                                   dtype=torch.bool)))

    def rope_graphsafe(self, q: torch.Tensor, k: torch.Tensor,
                       pos) -> Tuple[torch.Tensor, torch.Tensor]:
        """RoPE indexed by device position tensor `pos` (shape (N,)) — no host
        slicing, so capture-safe inside a CUDA Graph."""
        c = self.rope.cos.index_select(0, pos.long())   # (N, D)
        s = self.rope.sin.index_select(0, pos.long())
        c = c.unsqueeze(0)                               # (1, N, D)
        s = s.unsqueeze(0)
        D = q.size(-1)

        def rot(t):
            t1, t2 = t[..., :D // 2], t[..., D // 2:]
            return torch.cat([-t2, t1], dim=-1)
        q = q * c + rot(q) * s
        k = k * c + rot(k) * s
        return q, k

    def forward_static(self, x: torch.Tensor, pos, cache: dict) -> torch.Tensor:
        """CUDA-Graph-safe decode step. x: (B, N, D); pos: device int64 (N,).
        Writes new K,V into a static (B, L_fixed, d) buffer and attends over the
        full L_fixed width (invalid slots masked). Shapes are fixed given N."""
        B = x.size(0); N = x.size(1)
        L_fixed = self._L_fixed
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q, k = self.rope_graphsafe(q, k, pos)
        pos = pos.long()
        cache["k"][:, pos] = k          # in-place write N rows
        cache["v"][:, pos] = v
        K_all = cache["k"][:, :L_fixed]
        V_all = cache["v"][:, :L_fixed]
        attn = torch.bmm(q, K_all.transpose(1, 2)) / self.scale
        j = torch.arange(L_fixed, device=attn.device)
        allow = j[None, :] <= pos[:, None]            # (N, L_fixed)
        attn = attn.masked_fill(~allow.unsqueeze(0), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, V_all)
        return self.o_proj(out)


# ─────────────────────────────────────────────
#  Operator Router
# ─────────────────────────────────────────────
class OperatorRouter(nn.Module):
    """
    Scores: l_{i,v} = <proj(z_i), e_v>
    Training: Gumbel-Softmax top-k  (differentiable)
    Inference: hard top-k
    """
    def __init__(self, cfg: FNRConfig, K: int = None):
        super().__init__()
        d, d_r, k = cfg.d_model, cfg.d_router, cfg.k
        K = K if K is not None else cfg.K          # universal 共享库时用 shared_K
        self.proj = nn.Linear(d, d_r, bias=False)
        self.operator_embeds = nn.Parameter(
            torch.randn(K, d_r) * (1.0 / math.sqrt(d_r))
        )
        self.k = k
        self.K = K

    def forward(self, x: torch.Tensor, training: bool = True,
                tau: float = 5.0,
                noise: Optional[torch.Tensor] = None,
                bias: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            indices: (B, N, k)
            weights: (B, N, k)   (renormalised soft probabilities)
            full_dist: (B, N, K) (softmax over all K, for entropy reg)

        noise: 可选预生成 Gumbel 指数噪声 (B,N,K)，供 CUDA-Graph 捕获的训练
        使用 —— 把 RNG 解耦到图外，每次 replay 都能拿到新的噪声；传入 `noise`
        即替代图内会冻结的 exponential_ 抽取。
        """
        z = self.proj(x)                    # (B, N, d_r)
        scores = z.reshape(-1, z.size(-1)) @ self.operator_embeds.T
        scores = scores.reshape(x.shape[0], x.shape[1], -1)
        # 固定层偏置（universal）：0 参数、确定性 buffer，让第 li 层偏向共享库的
        # 不同段，保持层间差异而无需任何可训练变量（纯固定程序 + 固定偏移）。
        if bias is not None:
            scores = scores + bias
        full_dist = F.softmax(scores, dim=-1)

        B, N, K = scores.shape

        if training:
            if noise is not None:
                gumbel = -noise.float().log()          # device tensor, shape (B,N,K)
            else:
                gumbel = -torch.empty_like(scores).exponential_().log()
            soft_logits = (scores + gumbel) / tau
            soft_probs = F.softmax(soft_logits, dim=-1)
            topk_vals, topk_idx = torch.topk(soft_probs, self.k, dim=-1)
            weights = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)
        else:
            topk_vals, topk_idx = torch.topk(scores, self.k, dim=-1)
            weights = F.softmax(topk_vals / tau, dim=-1)

        return topk_idx, weights, full_dist


# ─────────────────────────────────────────────
#  FNR Block
# ─────────────────────────────────────────────
class FNRBlock(nn.Module):
    def __init__(self, cfg: FNRConfig, task_dir: Optional[torch.Tensor] = None,
                 is_mid_layer: bool = False,
                 shared_bank=None, shared_router=None, li: int = 0):
        super().__init__()
        freeze = bool(getattr(cfg, "freeze_norms", False))
        self.context_norm = RMSNorm(cfg.d_model, cfg.norm_eps, freeze=freeze)
        self.context_router = ContextRouter(cfg)

        self.operator_norm = RMSNorm(cfg.d_model, cfg.norm_eps, freeze=freeze)
        # universal 固定程序：注入共享路由控制器（1 份，L 层共用），
        # 并给本层注入固定的 0 参数层偏置（让不同层选库的不同段）。
        if shared_router is not None:
            self.operator_router = shared_router
        else:
            # 非共享模式：router 与本层独立 bank 的 cfg.K 对齐（绝不能用 shared_K）
            self.operator_router = OperatorRouter(cfg, K=cfg.K)
        # universal 固定程序：每层 1 个"固定层偏置"buffer（0 参数、确定性），
        # 让第 li 层偏向共享库的不同段，保持层间差异而无需任何可训练变量。
        if shared_router is not None or shared_bank is not None:
            K_eff = self.operator_router.K
            seg = li / max(1, cfg.n_layers)
            centre = int(seg * K_eff)
            width = max(1.0, K_eff / cfg.k)
            with torch.no_grad():
                bias = torch.exp(-((torch.arange(K_eff, dtype=torch.float32) - centre) ** 2)
                                 / (2.0 * width ** 2))
            bias = bias / (bias.max() + 1e-9) * 0.1
            self.register_buffer("layer_bias", bias.view(1, 1, K_eff), persistent=False)
        else:
            self.layer_bias = None
        if shared_bank is not None:
            self.operator_bank = shared_bank
        else:
            self.operator_bank = FrozenOperatorBank(
                K=cfg.K, d=cfg.d_model,
                lora_r=cfg.lora_r,
                activation=cfg.activation,
                lora_alpha=cfg.lora_alpha,
                frozen_fp16=getattr(cfg, "frozen_fp16", False),
                mb_budget=getattr(cfg, "gemm_mb_budget", 96),
                rank=getattr(cfg, "bank_rank", 0),
            )
        # Fix 1: coverage-constructive init of the frozen bank（全空间填充）
        if cfg.use_spectral_init:
            spectral_coverage_init(
                self.operator_bank,
                teacher_W=None,      # pass teacher FFN rows in production
                task_dir=task_dir,
                K_teacher=cfg.K_teacher_slots,
                K_anchor=cfg.K_anchor_slots,
                c_mix=cfg.c_mix,
            )
        # 低秩因子化（0 可训练变量，吞吐追 dense 的关键）：谱初始化在全空间
        # 完成后 SVD 截断成 Wa(d×r)/Wb(r×d)，运行期只 gather 两个小因子。
        if getattr(cfg, "bank_rank", 0) > 0 and self.operator_bank.rank > 0:
            self.operator_bank.factorize()
        # Fix 3: residual escape channel (parallel to frozen bank)
        self.escape = ResidualEscape(cfg.d_model, cfg.d_escape) \
            if cfg.use_residual_escape else None

        # 隐藏头零门控（"在隐藏层添加一个 0"）：中间层（除首尾）的整层
        # 残差增量（context + operator + escape 三项之和）乘以 1 个标量门控
        # g，g 零初始化 → 初始态中间层输出 == 输入（纯恒等，固定程序化），
        # 之前各层算出的 token 值不会被中间层的计算结果扰乱；训练时梯度
        # 推动 g 按需求打开。首尾层 g 恒为 1（不可训练），保持端点语义不变。
        if cfg.use_mid_zero_gate and is_mid_layer:
            self.mid_gate = nn.Parameter(torch.zeros(1))
        else:
            self.mid_gate = None

    def forward(self, x: torch.Tensor, training: bool = True,
                tau: float = 5.0, lora_scale: float = 0.0,
                start_pos: int = 0, noise: Optional[torch.Tensor] = None,
                route_bias: Optional[torch.Tensor] = None,
                chan_scale: Optional[torch.Tensor] = None):
        """
        Returns: (x_new, full_dist, z2, op_out, gate_abs)
        z2/op_out kept so the growth monitor can build residual covariance.
        gate_abs: mean |escape gamma| (or 0). mid layers additionally scale
        the whole residual delta by the zero-init scalar gate `mid_gate`.
        noise: 可选 Gumbel 噪声 (B,N,K)，CUDA-Graph 训练时由图外生成。
        """
        # Context sublayer
        z1 = self.context_norm(x)
        ctx_out = self.context_router(z1, start_pos)

        # Operator sublayer
        z2 = self.operator_norm(x + ctx_out)
        # 双引擎通道调整：遇到当前 token 时，按控制器值缩放各通道增益
        # （"调整模型所有通道需要在遇到什么 token 时改变路径连接的权重"）
        if chan_scale is not None:
            z2 = z2 * chan_scale
        # 双引擎路径调整：本层 top-k 路由偏置（固定 layer_bias + 控制器 path）
        bias = self.layer_bias
        if route_bias is not None:
            bias = bias + route_bias if bias is not None else route_bias
        indices, weights, full_dist = self.operator_router(z2, training, tau,
                                                           noise=noise,
                                                           bias=bias)
        op_out = self.operator_bank.forward_grouped(z2, indices, weights, lora_scale)

        # Fix 3: residual escape (zero at start; gate opens only when needed)
        delta = ctx_out + op_out
        gate_abs = x.new_zeros(())
        if self.escape is not None:
            delta = delta + self.escape(z2)
            if training:
                gate_abs = self.escape.gamma.detach().abs().mean()

        # 隐藏头零门控：中间层整层残差 delta 乘以 1 个 0-init 标量。
        # 初始态中间层输出 == 输入（纯恒等，固定程序化），已算出的 token
        # 值不会被中间层后续计算扰乱；训练时按需求自然打开。
        if self.mid_gate is not None:
            delta = delta * self.mid_gate

        x_new = x + delta
        return x_new, full_dist, z2, op_out, gate_abs

    def forward_cached(self, x: torch.Tensor, start_pos: int,
                       cache: dict, lora_scale: float = 0.0):
        """Inference fast-path: context sublayer uses KV cache (incremental),
        operator sublayer is pointwise (N new tokens, no cache needed).
        x: (B, N, d) with N usually = 1 during generation.
        Same zero-gate semantics as forward(): mid layers scale the whole
        residual delta by the zero-init scalar gate (fixed-program init)."""
        # Context sublayer (cached attention)
        z1 = self.context_norm(x)
        ctx_out = self.context_router.forward_cached(z1, start_pos, cache)

        # Operator sublayer (pointwise over the N new tokens)
        z2 = self.operator_norm(x + ctx_out)
        indices, weights, _full_dist = self.operator_router(z2, training=False,
                                                   bias=self.layer_bias)
        op_out = self.operator_bank.forward_grouped(z2, indices, weights, lora_scale)

        delta = ctx_out + op_out
        if self.escape is not None:
            delta = delta + self.escape(z2)
        if self.mid_gate is not None:
            delta = delta * self.mid_gate
        return x + delta

    def forward_static(self, x: torch.Tensor, pos, cache: dict,
                       lora_scale: float = 0.0) -> torch.Tensor:
        """CUDA-Graph-safe decode step: context sublayer uses the static KV
        buffer (position read from device tensor `pos`), operator sublayer is
        pointwise over the N new tokens. All shapes are fixed given N=B=1.
        Zero-gate semantics identical to forward() (graph-safe: gate is a
        scalar parameter, replay reads the current value on device)."""
        z1 = self.context_norm(x)
        ctx_out = self.context_router.forward_static(z1, pos, cache)

        z2 = self.operator_norm(x + ctx_out)
        indices, weights, _full_dist = self.operator_router(z2, training=False,
                                                   bias=self.layer_bias)
        op_out = self.operator_bank.forward_grouped(z2, indices, weights, lora_scale)

        delta = ctx_out + op_out
        if self.escape is not None:
            delta = delta + self.escape(z2)
        if self.mid_gate is not None:
            delta = delta * self.mid_gate
        return x + delta


# ─────────────────────────────────────────────
#  FNR-100M Model
# ─────────────────────────────────────────────
@dataclass
class FNRModelOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    routing_entropy: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None
    full_dists: Optional[List[torch.Tensor]] = None  # one per layer, each (B,N,K)


class FNRModel(nn.Module):
    def __init__(self, cfg: FNRConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embed.weight, std=0.02)
        L = cfg.n_layers
        # universal 固定程序：1 份共享冻结算子库 + 1 份共享路由控制器，
        # L 层共用（删掉 L 份独立算子库 / L 份控制器 embedding 的重复存储与搬运）。
        shared_bank, shared_router = None, None
        if getattr(cfg, "shared_library", False):
            shared_bank = FrozenOperatorBank(
                K=getattr(cfg, "shared_K", cfg.K), d=cfg.d_model,
                lora_r=0, activation=cfg.activation, lora_alpha=1.0,
                frozen_fp16=getattr(cfg, "frozen_fp16", False),
                # single_gemm：一次性处理更多 token = 提高 W_sel 预算（比默认 96MB 大），
                  # 减少分块 gather + 发射；但 2GB 卡 B16S256 全 batch 需 ~604MB live W_sel
                  # + autograd 双份 -> OOM，故预算取 cfg.single_gemm_mb（默认 192MB，
                  # chunk≈1352，比默认 676 翻倍，仍在 VRAM 内）。
                mb_budget=(getattr(cfg, "single_gemm_mb", 192)
                          if getattr(cfg, "single_gemm", False)
                          else getattr(cfg, "gemm_mb_budget", 96)),
            )
            if cfg.use_spectral_init:
                spectral_coverage_init(shared_bank, teacher_W=None, task_dir=None,
                                       K_teacher=cfg.K_teacher_slots,
                                       K_anchor=cfg.K_anchor_slots, c_mix=cfg.c_mix)
            if getattr(cfg, "bank_rank", 0) > 0:
                shared_bank.rank = getattr(cfg, "bank_rank", 0)
                shared_bank.factorize()
            shared_router = OperatorRouter(cfg, K=getattr(cfg, "shared_K", cfg.K))
        self.blocks = nn.ModuleList([
            FNRBlock(cfg, is_mid_layer=(cfg.use_mid_zero_gate and 0 < i < L - 1),
                     shared_bank=shared_bank, shared_router=shared_router, li=i)
            for i in range(L)
        ])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        # Tied head (Optimization #9)
        if cfg.tie_weights:
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            self.head.weight = self.embed.weight   # share parameter
        else:
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # ── 双引擎（3.5M 控制器 + 全冻结 50M）：先分词 → 0 变量函数计算器
        # 算出"需要放入模型中的值" → 该值初始化控制器 token 值表 → 控制器输出
        # 每 token 的"遇到什么 token 改哪条路径/哪个通道"。大引擎全程冻结。
        self.dual_engine = bool(getattr(cfg, "dual_engine", False))
        if self.dual_engine:
            # 1) 冻结大引擎：bank/embed/attn/router/norm/head 全部 requires_grad=False
            for p in self.parameters():
                p.requires_grad_(False)
            # 2) 函数计算器（0 可训练变量）：固定节点权重(embedding) × 正交基 → 定点签名
            self.func_calc = FunctionCalculator(
                vocab=cfg.vocab_size, d=cfg.d_model, C=cfg.dual_C,
                frozen_emb=self.embed.weight, seed=2026)
            # 3) 路径控制器（<=3.5M 可训练小引擎）：token 值表用计算器输出初始化
            self.path_controller = PathController(
                vocab=cfg.vocab_size, C=cfg.dual_C, L=cfg.n_layers,
                K=cfg.K, d=cfg.d_model, hidden=cfg.dual_hidden)
            self.path_controller.seed_from_calculator(self.func_calc)
            for p in self.path_controller.parameters():
                p.requires_grad_(True)
            # 4) 固定直通项（0 变量）：计算器值直接按固定系数调整通道/路径，
            #    从第 0 步就生效；可训练头只是其上的零初始化修正。
            gain = float(cfg.dual_fix_gain)
            self.register_buffer("dual_gain", torch.tensor(gain), persistent=False)
            self.register_buffer(
                "M_path", deterministic_orthonormal(cfg.n_layers * cfg.K, cfg.dual_C, seed=7),
                persistent=False)
            self.register_buffer(
                "M_chan", deterministic_orthonormal(cfg.d_model, cfg.dual_C, seed=11),
                persistent=False)

        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")
        # 图捕获训练用：tau 以设备侧 1 元素 buffer 形式喂给 forward，
        # 整个训练步被捕获成 CUDA Graph 后，tau 依然可以每步在 host 侧
        # 更新（fill_ 是异步 H2D 单值写，零同步），避免把 tau 烧死在图里。
        self.register_buffer("tau_buf", torch.full((1,), 5.0,
                                                    dtype=torch.float32),
                             persistent=False)

        # Fix 4: library growth monitor (dead-slot recycling + unexplained direction)
        self.growth_monitor: Optional[LibraryGrowthMonitor] = None
        if cfg.use_library_growth:
            self.growth_monitor = LibraryGrowthMonitor(
                L=cfg.n_layers, K=cfg.K, d=cfg.d_model,
                dead_frac=cfg.growth_dead_frac,
                actions_per_layer=cfg.growth_actions,
            )
        self._last_obs: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None
        # CUDA-Graph capture cache for generate_graph, keyed by the full
        # (L_fixed, temperature, top_k, do_sample) signature of the captured step.
        self._graph_cache: dict = {}

    def observe_growth(self, all_dists: List[torch.Tensor],
                       all_z2: List[torch.Tensor],
                       all_op: List[torch.Tensor]):
        """Feed per-layer (dist, z2, op_out) into the growth monitor."""
        if self.growth_monitor is None:
            return
        for l, (d, z, o) in enumerate(zip(all_dists, all_z2, all_op)):
            self.growth_monitor.observe(l, d, z, o)

    def apply_growth(self, step: int) -> int:
        """Recycle dead slots toward the unexplained direction.
        Returns number of slots recycled."""
        if self.growth_monitor is None:
            return 0
        actions = self.growth_monitor.propose()
        if not actions:
            return 0
        return LibraryGrowthMonitor.apply(
            self, actions, c_patch=self.cfg.growth_c_patch)

    # ── Progressive LoRA schedule ──
    @staticmethod
    def progressive_lora_scale(step: int, cfg: FNRConfig) -> float:
        """
        step < 2000:      0.0   (pure frozen, no LoRA)
        2000 <= step < 5000: linear 0→1
        step >= 5000:      full scale (lora_r as configured)
        """
        if not cfg.lora_progressive or cfg.lora_r == 0:
            return 1.0
        if step < 2000:
            return 0.0
        if step < 5000:
            return min(1.0, (step - 2000) / 3000.0)
        return 1.0

    # ── Adaptive entropy (Optimization #8) ──
    @staticmethod
    def adaptive_entropy_lambda(step: int, routing_entropy: float,
                                K: int, cfg: FNRConfig) -> float:
        """
        Base lambda = cfg.lambda_entropy.
        If entropy H < 0.5 * log(K) → collapse → boost lambda × 10.
        If H > 0.9 * log(K) → over-random → scale down to 0.5×.
        """
        import math
        logK = math.log(K)
        base = cfg.lambda_entropy
        if routing_entropy < 0.5 * logK:
            return base * 10.0
        elif routing_entropy > 0.9 * logK:
            return base * 0.5
        return base

    @staticmethod
    def adaptive_entropy_lambda_tensor(routing_entropy, K: int,
                                       base: float, device) -> torch.Tensor:
        """Device-side version: keeps the whole training step on-GPU, no .item() sync."""
        import math
        logK = math.log(float(K))
        low = torch.tensor(0.5 * logK, device=device)
        high = torch.tensor(0.9 * logK, device=device)
        boost = torch.tensor(base * 10.0, device=device)
        damp = torch.tensor(base * 0.5, device=device)
        norm = torch.tensor(base, device=device)
        return torch.where(routing_entropy < low, boost,
               torch.where(routing_entropy > high, damp, norm))

    def _ent_constants(self, device) -> Tuple[torch.Tensor, ...]:
        """Capture-safe constant tensors for the adaptive-entropy branches.
        Built ONCE on the device (so a CUDA-Graph-captured training step
        never performs an H2D tensor create inside the graph), then reused."""
        if not hasattr(self, "_ent_const") or getattr(self, "_ent_dev", None) != device:
            import math
            base = self.cfg.lambda_entropy
            logK = math.log(float(self.cfg.K))
            dev = device if isinstance(device, torch.device) else torch.device(device)
            self._ent_const = (
                torch.tensor(0.5 * logK, device=dev),
                torch.tensor(0.9 * logK, device=dev),
                torch.tensor(base * 10.0, device=dev),
                torch.tensor(base * 0.5, device=dev),
                torch.tensor(base, device=dev),
            )
            self._ent_dev = dev
        return self._ent_const

    def _adaptive_entropy_lambda_capture_safe(self, routing_entropy) -> torch.Tensor:
        low, high, boost, damp, norm = self._ent_constants(routing_entropy.device)
        return torch.where(routing_entropy < low, boost,
                           torch.where(routing_entropy > high, damp, norm))

    def get_tau(self, step: int) -> float:
        t = min(1.0, step / self.cfg.tau_anneal_steps)
        return self.cfg.tau_start + (self.cfg.tau_end - self.cfg.tau_start) * t

    # ── Forward ──
    def forward(self, input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                training: bool = True, step: int = 0,
                lora_scale_override: Optional[float] = None,
                tau: Optional[torch.Tensor] = None,
                noise: Optional[torch.Tensor] = None,
                ) -> FNRModelOutput:
        B, N = input_ids.shape
        x = self.embed(input_ids)
        if tau is None:
            tau = self.get_tau(step)
        lora_scale = lora_scale_override if lora_scale_override is not None \
                     else self.progressive_lora_scale(step, self.cfg)

        all_full_dists: List[torch.Tensor] = []
        all_z2: List[torch.Tensor] = []
        all_op: List[torch.Tensor] = []
        # 双引擎：一次前调控制器 + 固定直通项（计算器值 × 固定正交基）
        dual_path = dual_chan = None
        if self.dual_engine:
            ctrl = self.path_controller(input_ids)
            sig = self.func_calc.sig.index_select(0, input_ids.reshape(-1)) \
                .view(B, N, self.cfg.dual_C)
            path_fix = self.dual_gain * (sig @ self.M_path.T) \
                .view(B, N, self.cfg.n_layers, self.cfg.K)
            chan_fix = self.dual_gain * (sig @ self.M_chan.T)
            dual_path = ctrl["path"] + path_fix      # 可训练修正 + 固定直通
            dual_chan = ctrl["chan"] + chan_fix
        for li, block in enumerate(self.blocks):
            nl = noise[li] if noise is not None else None
            rb = dual_path[:, :, li, :] if dual_path is not None else None
            cs = dual_chan if dual_chan is not None else None
            x, full_dist, z2, op_out, _gate = block(
                x, training=training, tau=tau, lora_scale=lora_scale,
                noise=nl, route_bias=rb, chan_scale=cs)
            all_full_dists.append(full_dist)
            all_z2.append(z2)
            all_op.append(op_out)

        # Fix 4: feed per-layer stats into the library-growth monitor
        if training and self.growth_monitor is not None:
            self.observe_growth(all_full_dists, all_z2, all_op)

        x = self.final_norm(x)
        logits = self.head(x)   # (B, N, vocab)

        # Routing entropy
        stacked = torch.stack(all_full_dists, dim=0)   # (L, B, N, K)
        routing_entropy = -(stacked * torch.log(stacked + 1e-10)).sum(-1).mean()

        out = FNRModelOutput(
            logits=logits,
            routing_entropy=routing_entropy,
            full_dists=all_full_dists,
        )

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            token_loss = self.loss_fn(shift_logits.reshape(-1, self.cfg.vocab_size),
                                      shift_labels.reshape(-1))

            # Adaptive entropy regularisation — device-side (no .item() sync);
            # capture-safe constant tensors so it works inside a CUDA-Graph step.
            ent_lambda = self._adaptive_entropy_lambda_capture_safe(routing_entropy)
            ent_loss = -routing_entropy   # minimise → maximise entropy
            load_bal = cfg_load_balance(stacked, self.cfg.lambda_load_bal, self.cfg.K)

            aux = ent_lambda * ent_loss + load_bal
            out.aux_loss = aux
            out.loss = token_loss + aux

        return out

    # ── Generation (KV-cache accelerated: O(N) decode, not O(N^2)) ──
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor,
                 max_new_tokens: int = 64,
                 temperature: float = 0.8,
                 top_k: int = 50,
                 do_sample: bool = True) -> torch.Tensor:
        self.eval()
        B = input_ids.size(0)
        lora_scale = self.progressive_lora_scale(0, self.cfg)
        caches = [{} for _ in self.blocks]

        def final_logits(hidden_last):
            """hidden_last: (B, d) final-layer hidden -> (B, vocab) logits."""
            h = self.final_norm(hidden_last.unsqueeze(1))
            return self.head(h).squeeze(1)

        # 1) prefill the prompt in one shot (P new tokens, build all KV caches)
        pre = input_ids[:, -self.cfg.max_seq_len:]
        P = pre.size(1)
        x = self.embed(pre)                                  # (B, P, d)
        for li, block in enumerate(self.blocks):
            x = block.forward_cached(x, start_pos=0, cache=caches[li],
                                     lora_scale=lora_scale)
        gen_ids = pre.clone()
        start_pos = P

        # 2) incremental decode: each step runs ONE token through all layers
        #    with persistent KV (O(N) attention instead of O(N^2)).
        for _ in range(max_new_tokens):
            logits = final_logits(x[:, -1, :])               # predict next token
            logits = logits / temperature
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                v, idx = torch.topk(logits, k, dim=-1)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, idx, v)
                logits = mask
            probs = F.softmax(logits, dim=-1)
            nxt = (torch.multinomial(probs, num_samples=1) if do_sample
                   else logits.argmax(dim=-1, keepdim=True))
            gen_ids = torch.cat([gen_ids, nxt], dim=1)
            if (nxt == self.cfg.eos_token_id).all():
                break
            # advance: run the new token through all layers (incremental)
            x_new = self.embed(nxt)                          # (B, 1, d)
            for li, block in enumerate(self.blocks):
                x_new = block.forward_cached(x_new, start_pos=start_pos,
                                             cache=caches[li], lora_scale=lora_scale)
            x = x_new
            start_pos += 1
        return gen_ids

    # ── Generation with CUDA Graphs (theoretical-limit inference) ──
    @torch.no_grad()
    def generate_graph(self, input_ids: torch.Tensor,
                      max_new_tokens: int = 64,
                      temperature: float = 0.8,
                      top_k: int = 50,
                      do_sample: bool = False,
                      max_len: int = 512) -> torch.Tensor:
        """
        CUDA-Graph accelerated decode (B=1). Each decode step is a fixed shape
        (B=1, N=1): embed(carried token) -> 12 blocks (static KV) -> head ->
        sample. The whole step is captured into ONE graph, so the ~12-layer
        per-token launch overhead collapses to a single graph replay. This is
        the tool that actually reaches the theoretical limit on Pascal, where
        torch.compile/Triton is unavailable and the bank GEMM is launch-bound.
        NOTE: do_sample uses multinomial which IS capture-safe; greedy is the
        common case.
        """
        assert input_ids.size(0) == 1, "generate_graph supports B=1"
        self.eval()
        dev = input_ids.device
        L_fixed = max_len
        lora_scale = self.progressive_lora_scale(0, self.cfg)
        temp_key = round(float(temperature), 3)
        gkey = (L_fixed, temp_key, int(top_k), bool(do_sample))

        def final_logits(h11):
            return self.head(self.final_norm(h11)).squeeze(1)

        # ---- prefill the prompt (fills the entry's static KV up to P) ----
        pre = input_ids[:, -self.cfg.max_seq_len:]
        P = pre.size(1)

        # Build (or reuse) the static buffers + captured decode graph.
        entry = self._graph_cache.get(gkey)
        if entry is None:
            static = [{} for _ in self.blocks]
            for li, blk in enumerate(self.blocks):
                blk.context_router._init_static(L_fixed, dev)
                d_r, d_v = self.cfg.d_router, self.cfg.d_v
                static[li]["k"] = torch.zeros(1, L_fixed, d_r, device=dev)
                static[li]["v"] = torch.zeros(1, L_fixed, d_v, device=dev)
            pos = torch.tensor([P], device=dev, dtype=torch.long)
            tok_in = torch.zeros(1, 1, device=dev, dtype=torch.long)

            # capture a single fixed-shape decode step
            def one_decode_captured():
                h = self.embed(tok_in)
                for li, blk in enumerate(self.blocks):
                    h = blk.forward_static(h, pos, static[li],
                                           lora_scale=lora_scale)
                lg = final_logits(h) / temperature
                if top_k > 0:
                    kk = min(top_k, lg.size(-1))
                    vv, ix = torch.topk(lg, kk, dim=-1)
                    mm = torch.full_like(lg, float("-inf"))
                    mm.scatter_(1, ix, vv)
                    lg = mm
                pr = F.softmax(lg, dim=-1)
                nxt = (torch.multinomial(pr, num_samples=1) if do_sample
                       else lg.argmax(-1, keepdim=True))
                tok_in.copy_(nxt)
                pos.add_(1)

            for _ in range(2):
                one_decode_captured()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                one_decode_captured()
            torch.cuda.synchronize()
            entry = {"graph": g, "static": static, "pos": pos, "tok_in": tok_in}
            self._graph_cache[gkey] = entry

        static = entry["static"]; pos = entry["pos"]; tok_in = entry["tok_in"]
        g = entry["graph"]

        # fresh prompt-specific prefill into the shared static buffers
        # (use forward_static so zero-gate semantics stay in ONE place)
        x = self.embed(pre)
        pos_pre = torch.arange(P, device=dev)
        for li, blk in enumerate(self.blocks):
            x = blk.forward_static(x, pos_pre, static[li], lora_scale=lora_scale)
        first_tok = final_logits(x[:, -1, :]).argmax(-1, keepdim=True)

        # reset the graph-mutable state to this prompt, then replay
        pos.fill_(P)
        tok_in.copy_(first_tok)

        gen_ids = pre.clone()
        for i in range(max_new_tokens):
            gen_ids = torch.cat([gen_ids, tok_in], dim=1)
            if (tok_in == self.cfg.eos_token_id).all():
                break
            if i < max_new_tokens - 1:
                g.replay()
        return gen_ids

    # ── Parameter counts ──
    def count_params(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # 共享库模式下 L 个 block 引用同一份 FrozenOperatorBank，按 id 去重，
        # 避免把同一份"固定程序"计 L 遍（那是 1 份数据，不是 L 份数据）。
        bank_frozen = 0
        seen = set()
        for block in self.blocks:
            bank = block.operator_bank
            if id(bank) in seen:
                continue
            seen.add(id(bank))
            # 低秩模式 bank.W 已被 factorize() 删除，改用 frozen_numel() 统一计数
            bank_frozen += bank.frozen_numel()
        if getattr(self, "dual_engine", False):
            # 双引擎：50M 大引擎全冻结（bank + embed/attn/router/norm/head），
            # 可训练核只有 PathController（<=3.5M 预算）。
            frozen_other = sum(p.numel() for p in self.parameters()
                              if not p.requires_grad)
            frozen = bank_frozen + frozen_other
            total = trainable + frozen
            return {
                "total": total,
                "trainable": trainable,
                "frozen": frozen,
                "frozen_banks": bank_frozen,
                "frozen_other": frozen_other,
                "controller": trainable,
                "ratio": f"{frozen // max(trainable, 1)}:1",
            }
        frozen = bank_frozen
        total = trainable + frozen
        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "ratio": f"{frozen // max(trainable, 1)}:1",
        }


def cfg_load_balance(dists: torch.Tensor, lam: float, K: int = 1) -> torch.Tensor:
    """Standard MoE auxiliary: penalise variance across operators."""
    p_v = dists.mean(dim=(0, 1, 2))        # (K,)
    return lam * K * (p_v ** 2).sum()
