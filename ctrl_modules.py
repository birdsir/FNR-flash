"""
双引擎预训练控制核（"3.5M 控制器驱动全冻结 50M 模型"）。

思想（按用户指令落地，不实测）：
  1) 数据先分词（复用 tokenizer/prep_data，token 值就是输入）；
  2) 把「token 值 × 固定在神经节点里的权重值」放入 **FunctionCalculator**
     （0 可训练变量）中，算出"需要放入模型中的值"（每 token 的固定通道签名）；
  3) **PathController**（可训练小核，≤3.5M 预算）用该值初始化自己的 token 值表，
     并输出「模型所有通道在遇到这个 token 时应当改变的路径连接权重」：
       - 每层 top-k 路由偏置 path (L,K)：遇到该 token 时改哪条路径；
       - 通道缩放 chan (d)：各通道连接权重增益；
  4) 50M 参数模型全部冻结（bank/embed/attn/router/norm/head），
     优化器只更新控制器 —— "用 3.5M 参数训练一个 50M 模型"。

初始态保证（与 LoRA 零初始化惯例一致）：
  path_head / chan_head 零初始化 ⇒ 控制器增量恒 0，
  模型从第 0 步起 = 100% 固定程序（计算器值 + 冻结 50M），
  训练才逐渐把"遇到什么 token 改哪条路径"打开。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def deterministic_orthonormal(rows: int, cols: int, seed: int = 2026,
                               device="cpu") -> torch.Tensor:
    """固定、可复现的正交基（0 可训练变量，种子确定性 —— 属于'固定程序'）。"""
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(rows, cols, generator=g, device=device)
    if rows >= cols:
        q, _ = torch.linalg.qr(raw)
        return q[:, :cols]
    q, _ = torch.linalg.qr(raw.t())
    return q.t()


class FunctionCalculator(nn.Module):
    """0 可训练变量的'函数计算器'：把 token 值 + 固定节点权重值 算成
    '需要放入模型中的值'（每 token 固定通道签名）。

    f(v) = round16( normalize( emb_frozen(v) @ M_fixed^T ) / ||·|| )
      emb_frozen : 模型冻结 embedding 的一行（固定节点权重值之一）
      M_fixed    : C×d 正交基（种子确定性，0 可训练变量）
      round16    : 定点化（量化到 1/16），强调'固定值表'而非连续变量

    输出是纯函数值（buffer 表，vocab×C），不参与梯度，不占 3.5M 预算。
    """
    def __init__(self, vocab: int, d: int, C: int,
                 frozen_emb: torch.Tensor, seed: int = 2026):
        super().__init__()
        self.C = C
        M = deterministic_orthonormal(C, d, seed=seed).to(frozen_emb.device)
        with torch.no_grad():
            emb = frozen_emb.detach().cpu().float()          # (vocab, d) 固定值
            sig = emb @ M.T                                    # (vocab, C)
            sig = sig / sig.norm(dim=1, keepdim=True).clamp_min(1e-9)
            sig = torch.round(sig * 16.0) / 16.0               # 定点化
        self.register_buffer("sig", sig.to(frozen_emb.device).to(frozen_emb.dtype),
                             persistent=False)

    def values(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: 任意形状 long → sig[token_ids]，末维 C。"""
        flat = self.sig.index_select(0, token_ids.reshape(-1))
        return flat.view(*token_ids.shape, self.C)

    def extra_repr(self):
        return f"C={self.C}, table={self.sig.shape[0]}x{self.C} (0 trainable)"


class PathController(nn.Module):
    """可训练路径控制器（≤3.5M 参数预算），输出'所有通道在遇到该 token 时
    改变的路径连接权重'。

    可训练组成（50M 预设 d=160, L=12, K=156, C=224 时 ≈ 2.4M）：
      tok_table : (vocab, C)  token 值表 —— 由 FunctionCalculator 初始化
                  （'函数计算器算出的值放入模型'即此表的初值）；
      trunk     : Linear(C,h)+GELU+Linear(h,C) 残差细化；
      path_head : Linear(C, L*K) 零初始化 → 每层 top-k 路由偏置；
      chan_head : Linear(C, d)  零初始化 → 每 token 通道缩放。

    零初始化头 ⇒ 初始增量恒 0：模型 100% 固定程序，训练才逐渐打开。
    固定直通项（0 变量，在 FNRModel 侧计算）：
      path = head(h) + gain·sig@M_path   chan = head(h) + gain·sig@M_chan
    即'计算出的值'从第 0 步起就按固定系数调整通道，可训练头只做修正。
    """
    def __init__(self, vocab: int, C: int, L: int, K: int, d: int,
                 hidden: int = 320):
        super().__init__()
        self.C, self.L, self.K, self.d = C, L, K, d
        self.tok_table = nn.Parameter(torch.empty(vocab, C))
        self.trunk1 = nn.Linear(C, hidden, bias=True)
        self.trunk2 = nn.Linear(hidden, C, bias=True)
        self.path_head = nn.Linear(C, L * K, bias=False)
        self.chan_head = nn.Linear(C, d, bias=False)
        nn.init.zeros_(self.path_head.weight)   # 初始 = 纯固定程序
        nn.init.zeros_(self.chan_head.weight)

    def seed_from_calculator(self, calc: FunctionCalculator):
        """把函数计算器算出的'需要放入模型的值'作为 token 值表初值。"""
        with torch.no_grad():
            self.tok_table.copy_(calc.sig.to(self.tok_table.dtype))

    def forward(self, token_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """token_ids (…, N) → {'path': (…, N, L, K), 'chan': (…, N, d)}。"""
        z = self.tok_table.index_select(0, token_ids.reshape(-1))
        z = z.view(*token_ids.shape, self.C)
        h = F.gelu(self.trunk1(z))
        h = self.trunk2(h) + z                 # 残差
        B_shape = (*token_ids.shape, self.L, self.K)
        return {
            "path": self.path_head(h).view(B_shape),
            "chan": self.chan_head(h),
        }

    def extra_repr(self):
        n = sum(p.numel() for p in self.parameters())
        return f"C={self.C}, L*K={self.L * self.K}, d={self.d}, params={n / 1e6:.2f}M"
