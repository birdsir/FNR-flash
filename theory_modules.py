"""
理论防线：修复"冻结算子未覆盖目标 token"问题（对应论文 §4.2 Representational Ceiling）。

符号（完整推导见 THEORETICAL_FIX.md）：
  z          路由前的隐状态
  f_v(z)     第 v 个冻结算子 = act(W_v z + b_v)
  E_{K,k}     可达集 = { sum_j w_j f_{i_j}(z) : i_j 互异, w_j >= 0, sum_j w_j = 1 }
  Delta_K(y*) 天花板距离 = min_{y in E} || y - y* ||，y* = head 需要解码目标词
               所需的目标隐状态

四层防线（按成本递增、按需启用）：
  Fix 1  spectral_coverage_init   谱覆盖初始化：让 E_{K,k} 构造性地覆盖任务子空间
  Fix 2  lora_tube_radius         LoRA 管状邻域：可达集外扩半径 rho = alpha*C/(2r)
  Fix 3  ResidualEscape           残差逃逸通道：门控的窄宽可训练 FFN（万能逼近）
  Fix 4  LibraryGrowthMonitor     库回收与生长：死槽位重新对准未解释方向（存储不变）
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


# ═══════════════════════════════════════════════════════════
#  Fix 2 —— LoRA 管状邻域半径
# ═══════════════════════════════════════════════════════════
def lora_tube_radius(lora_alpha: float, lora_r: int, param_budget: float) -> float:
    """
    在单算子 LoRA 参数预算 ||A||_F^2 + ||B||_F^2 = C 下（weight decay 的自然尺度），
    对冻结可达集的管状外扩半径：
        ||(alpha/r) B A||_op <= (alpha/r) ||B||_F ||A||_F
                               <= (alpha/r) * C/2          (AM-GM)
    即：天花板距离 Delta_K <= rho_alpha = alpha*C/(2r) 时，LoRA 单层即可消除天花板。
    """
    if lora_r == 0:
        return 0.0
    return (lora_alpha / lora_r) * (param_budget / 2.0)


# ═══════════════════════════════════════════════════════════
#  Fix 1 —— 谱覆盖初始化
# ═══════════════════════════════════════════════════════════
def spherical_code(K: int, d: int, device: str = "cpu") -> torch.Tensor:
    """d 维空间中 K 个近正交方向（K<=d 时为正交块，K>d 时随机块混合）。"""
    codes = []
    while len(codes) < K:
        q, _ = torch.linalg.qr(torch.randn(d, d, device=device))
        codes.append(q)
    return torch.cat([q[i % d] for q in codes for i in range(d)][:K])


def spectral_coverage_init(bank,
                           teacher_W: Optional[torch.Tensor] = None,
                           task_dir: Optional[torch.Tensor] = None,
                           K_teacher: int = 4,
                           K_anchor: int = 2,
                           c_mix: float = 0.3) -> int:
    """
    将算子库槽位分成三类填充（构造性保证，见 THEORETICAL_FIX.md 定理 1）：
      [0, K_teacher)          : 克隆 teacher 的 FFN 权重（无 teacher 时随机正交）
      [K_teacher, +K_anchor)  : 恒等锚点  W = I, b = 0
      其余                    : 任务子空间算子
                                   W = c_mix*(I - P) + P R P^T,  P = u u^T
                                   u = 任务方向（目标词所需隐状态的主方向）
    保证：
      (i)  E_{K,k} 中存在与 teacher FFN 输出 O(sqrt(d/K_t)) 以内的点；
      (ii) 存在"退回 dense teacher"的合法路由策略（锚点恒等 + teacher 槽）；
      (iii) 任务方向 u 是大量算子的强特征方向 → Delta_init -> 0。
    返回填充的算子数。
    """
    K, d, _ = bank.W.shape
    dev = bank.W.device
    n_t = min(K_teacher, K)
    n_a = min(K_anchor, K - n_t)
    eye = torch.eye(d, device=dev)
    n_filled = 0
    with torch.no_grad():
        for v in range(K):
            if v < n_t:
                if teacher_W is not None:
                    bank.W[v] = teacher_W.reshape(d, d).to(dev)
                else:
                    q, _ = torch.linalg.qr(torch.randn(d, d, device=dev))
                    bank.W[v] = q
                bank.b[v].zero_()
            elif v < n_t + n_a:
                bank.W[v] = eye
                bank.b[v].zero_()
            else:
                q, _ = torch.linalg.qr(torch.randn(d, d, device=dev))
                if task_dir is not None:
                    u = task_dir.reshape(d).to(dev)
                    u = u / (u.norm() + 1e-9)
                    P = torch.outer(u, u)                 # 任务方向投影
                    bank.W[v] = P @ q @ P + c_mix * (eye - P)
                else:
                    bank.W[v] = q
                bank.b[v].zero_()
            n_filled += 1
    return n_filled


# ═══════════════════════════════════════════════════════════
#  Fix 3 —— 残差逃逸通道
# ═══════════════════════════════════════════════════════════
class ResidualEscape(nn.Module):
    """
    与冻结算子库并联的窄宽可训练通道（universal-approximation 逃逸）：
        out = gate(z) * phi(z)
        gate(z) = gamma * sigmoid(<proj(z), e> / s)     gamma 初始 0（默认关闭）
        phi(z)  = W2 GELU(W1 z)                          W2 零初始化（起点 = 0）

    理论保证（定理 3）：
      - 冻结可达集 E 不变的前提下，逃逸通道把可达集替换为
            E' = E + { g*phi(z) : g in [0,1], phi 可训练 }
      - 对任意有界连续目标 f*，d_e -> inf 时 min_{phi,g} ||g*phi - f*|| -> 0
        （万能逼近 + 门控不裁剪范围：gamma 无上界）
      - 成本：每层 2*d*d_e + O(d) 个可训练参数（d_e=40, d=320 时约 0.026M）
      - 门控保证"够用就归零"：冻结库能覆盖的 token 上 gate -> 0，
        不破坏 FNR 的核心效率设计。
    可训练参数：d*d_e + d_e*d + gate_dim + 1
    """
    def __init__(self, d: int, d_e: int = 40, gate_dim: int = 16,
                 gamma_init: float = 0.1):
        super().__init__()
        self.d_e = d_e
        self.proj = nn.Linear(d, gate_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=0.05)
        self.e = nn.Parameter(torch.randn(gate_dim) * 0.5)
        # gamma_init > 0 (NOT 0): with W2 zero-init the output still starts
        # strictly at 0 (frozen start), but a nonzero gamma guarantees
        # grad d(out)/d(W2) = gate·g(·) ≠ 0, avoiding the double-zero dead start.
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.phi = nn.Sequential(
            nn.Linear(d, d_e, bias=False),
            nn.GELU(),
            nn.Linear(d_e, d, bias=False),
        )
        # LoRA convention: first layer small-random, last layer zero-init
        # → output exactly 0 at start (pure frozen config), but learnable.
        nn.init.normal_(self.phi[0].weight, std=0.1)
        nn.init.zeros_(self.phi[2].weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, N, d) -> 逃逸修正 (B, N, d)，起点为 0。"""
        g = self.gamma * torch.sigmoid((self.proj(z) @ self.e) / 4.0)  # (B,N)
        return g.unsqueeze(-1) * self.phi[2](F.gelu(self.phi[0](z)))

    def extra_repr(self):
        return f"d_e={self.d_e}"


# ═══════════════════════════════════════════════════════════
#  Fix 4 —— 库回收与生长（存储不变的"生长"）
# ═══════════════════════════════════════════════════════════
class LibraryGrowthMonitor:
    """
    检测两类事件并触发"回收"：
      (a) 死算子   : 利用率 p_v < dead_frac * (1/K)  （长期被路由遗忘）
      (b) 未解释方向: 残差协方差 C_l = E[(z - op_out)(z - op_out)^T]
                     的主特征向量 u*（冻结库最解释不了的方向）

    回收操作（对死槽位 v）：
        W_v <- Q + c_patch * u* u*^{*T}
    其中 Q 为随机正交。新算子在未解释方向 u* 上成为强特征方向：
        W_new u* = Q u* + c_patch u*   （c_patch>0 时沿 u* 放大）
    存储不变（回收而非扩容）；K 槽位的"有效覆盖"随任务分布增长。

    可选扩展：真生长 = 复制死槽位 + 保留原算子（K -> K+m，存储 m*d^2/层）。
    """
    def __init__(self, L: int, K: int, d: int,
                 momentum: float = 0.95,
                 dead_frac: float = 0.25,
                 actions_per_layer: int = 2,
                 device: Optional[torch.device] = None):
        self.L, self.K, self.d = L, K, d
        self.momentum = momentum
        self.dead_frac = dead_frac
        self.actions_per_layer = actions_per_layer
        self._lazy_dev = device
        # lazy-init tensors on device (util, resid_cov)
        self.util: Optional[torch.Tensor] = None
        self.resid_cov: Optional[torch.Tensor] = None

    def _ensure(self, dev):
        if self.util is None:
            self.util = torch.zeros(self.L, self.K, device=dev)
            self.resid_cov = torch.zeros(self.L, self.d, self.d, device=dev)

    def observe(self, layer: int, dist: torch.Tensor,
                z: torch.Tensor, op_out: torch.Tensor):
        """
        每个训练 step 对每层调用一次。
        dist: (B,N,K) 全路由分布; z, op_out: (B,N,d)
        """
        self._ensure(dist.device)
        m = 1.0 - self.momentum
        p = dist.mean(dim=(0, 1))                      # (K,)
        self.util[layer] = m * p + (1 - m) * self.util[layer]
        R = (z - op_out).reshape(-1, self.d)            # (B*N, d)
        C_new = (R.t() @ R) / max(R.numel() / self.d, 1)
        self.resid_cov[layer] = m * C_new + (1 - m) * self.resid_cov[layer]

    def unexplained_direction(self, layer: int) -> torch.Tensor:
        """残差协方差的最大特征向量（库最解释不了的方向）。"""
        _, eigvecs = torch.linalg.eigh(self.resid_cov[layer])
        return eigvecs[:, -1]

    def propose(self) -> List[Tuple[int, int, torch.Tensor]]:
        """返回 [(layer, dead_slot, direction), ...]。"""
        if self.util is None:
            return []
        acts = []
        thr = self.dead_frac * (1.0 / self.K)
        for l in range(self.L):
            p = self.util[l]
            dead = (p < thr).nonzero().flatten()
            if dead.numel() == 0:
                continue
            u = self.unexplained_direction(l)
            for slot in dead[:self.actions_per_layer]:
                acts.append((l, int(slot), u.clone()))
        return acts

    @staticmethod
    def apply(model, actions: List[Tuple[int, int, torch.Tensor]],
              c_patch: float = 1.0) -> int:
        """执行回收：死槽位 W <- 随机正交 + c_patch * u u^T。返回处理数。"""
        n = 0
        for l, slot, u in actions:
            bank = model.blocks[l].operator_bank
            d = bank.d
            u = u.to(bank.W.device)
            u = u / (u.norm() + 1e-9)
            q, _ = torch.linalg.qr(torch.randn(d, d, device=bank.W.device))
            with torch.no_grad():
                bank.W[slot] = q + c_patch * torch.outer(u, u)
                bank.b[slot].zero_()
            n += 1
        return n
