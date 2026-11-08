"""
FNR-100M: Optimized Fixed-Node Routing Network
Total ~100M parameters, designed for Chinese language modeling.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FNRConfig:
    # === Model dimensions ===
    d_model: int = 320          # hidden dimension
    n_layers: int = 12          # number of FNR blocks
    label: str = "100M"          # preset tag for summary() header
    K: int = 76                 # frozen operators per layer
    k: int = 4                  # top-k routing budget (increased from 2 → 4)
    d_router: int = 64          # router embedding dim (<< d_model)
    n_heads: int = 8            # attention heads
    head_dim: int = 40          # d_model / n_heads

    # === Vocab & head ===
    vocab_size: int = 8192      # Chinese subword (BPE/SentencePiece)
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    max_seq_len: int = 1024

    # === Optimizations (vs original paper) ===
    # 1. Value projection: was identity → now trainable (d_v × d)
    d_v: int = 64               # value projection dim
    # 2. LoRA: progressive low-rank adaptation
    lora_r: int = 2            # LoRA rank (0 = disabled, start at 0, grow to r)
    lora_alpha: float = 1.0
    lora_progressive: bool = True  # enable at step 2000, full at step 5000
    # 3. Activation: GELU instead of Sigmoid
    activation: str = "gelu"
    # 冻结算子低秩因子化（吞吐优化，0 可训练变量）：bank_rank=0 存全 D×D；
    # >0 时把每个冻结算子存成 Wa(K,d,r)·Wb(K,r,d) 两个小因子，运行期 gather
    # 字节从 k·d² 降到 2·k·d·r，且把大 batched GEMM 拆成两个宽 M 小 GEMM。
    # 诚实代价：每个算子表达力封顶在秩 r（d=96 时 r=24 ≈ 保留 ~50% 自由度）。
    bank_rank: int = 0
    # 4. Norm: RMSNorm instead of LayerNorm
    norm_eps: float = 1e-5
    # 5. Tied embedding/head
    tie_weights: bool = True
    # 6. Temperature annealing
    tau_start: float = 5.0
    tau_end: float = 0.5
    tau_anneal_steps: int = 5000
    # 7. Entropy regularization
    lambda_entropy: float = 0.05
    lambda_load_bal: float = 0.01
    # 8. RoPE
    use_rope: bool = True
    rope_theta: float = 10000.0

    # === Theoretical ceiling fixes (THEORETICAL_FIX.md) ===
    use_spectral_init: bool = True     # Fix 1: coverage-constructive init
    K_teacher_slots: int = 4           # slots cloned from teacher FFN
    K_anchor_slots: int = 2            # identity anchors (fallback = dense teacher)
    c_mix: float = 0.3                 # task-direction mixing coefficient
    use_residual_escape: bool = True   # Fix 3: gated narrow trainable channel
    d_escape: int = 40                 # escape hidden dim
    use_library_growth: bool = False   # Fix 4: recycle dead slots
    growth_interval: int = 1000        # propose/apply cycle (steps)
    growth_dead_frac: float = 0.25
    growth_actions: int = 2
    growth_c_patch: float = 1.0

    # === Training ===
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 20000
    batch_size: int = 32
    grad_clip: float = 1.0
    # Pascal GPUs (GP10x, no Tensor Core) are FASTER in fp32 than fp16 autocast.
    # Default off; train.py auto-benches both and picks the faster one.
    fp16: bool = False
    # 隐藏头(中间层)零门控：除首尾层外，每个 block 的残差增量乘以 1 个
    # 零初始化标量门控（“添加一个 0”）。初始态中间层 = 恒等映射，
    # 不会把已算好的 token 值扰乱；训练中门控按需求自然打开。
    # 这是固定程序化（1 个乘法）+ 极少变量（L-2 个标量，8M 预设共 8 个）。
    use_mid_zero_gate: bool = False
    # 整步 forward+backward 捕获为 CUDA Graph（“固定程序化”训练）：
    # 数据/噪声/tau 走设备侧静态 buffer，消除 10 层 × fwd+bwd 的启动开销。
    # 仅 fp32 + 固定 (B,S) 形状时启用；本卡上是最接近理论极限的训练提速。
    train_cuda_graph: bool = False
    # 冻结算子库 GEMM 走 fp16（Pascal 消费卡 FP16 FLOPs = 2×FP32，
    # 且 W_sel gather 字节减半）。固定程序化：算子本身不训练，fp16 对
    # 冻结计算只是确定性舍入（≈量化噪声），不改变 1 个可训练变量。
    # 仅当 lora_r==0 时生效（LoRA 增量保持 fp32 精度）。
    frozen_fp16: bool = False
    # VRAM 自适应分块 GEMM 的 W_sel 预算（MB），0 = 96MB 默认
    gemm_mb_budget: int = 96

    # ── Universal fixed-program limit (多固定、极少变量 → "万能程序") ──
    # 把"固定程序"推到极限：全网络共享 1 份冻结算子库 + 1 份路由控制器，
    # 层与层只靠固定的 0 参数"层模板"区分调用；可训练核心只剩"如何路由"。
    # 同时删除重复数据搬运（L 份库→1 份；L 份控制器→1 份；L 份 norm→冻结）、
    # 单次大 GEMM 一次性处理整 batch（删分块重复 gather + 发射）。
    shared_library: bool = False     # 跨层共享 1 份冻结算子库（万能程序体）
    shared_router: bool = False      # 跨层共享 1 份路由控制器
    shared_K: int = 400             # 共享库大小（比独立 K 更大，覆盖更广）
    freeze_norms: bool = False       # 冻结 RMSNorm 权重（全 1，删 L 份重复变量）
    single_gemm: bool = False        # 单次 GEMM 处理整 batch（提 W_sel 预算）
    single_gemm_mb: int = 192         # single_gemm 开启时的 W_sel 预算(MB)；2GB 卡 B16S256 实测 192MB 安全(约 604MB 全 batch 会 OOM)
    # ── 双引擎预训练（3.5M 控制器驱动全冻结 50M，见 DUAL_ENGINE_50M.md）──
    # 思想：先分词 -> token 值 + 固定节点权重 放入 0 变量 FunctionCalculator
    # 算出"需要放入模型中的值" -> 该值初始化 PathController 的 token 值表 ->
    # 控制器（<=3.5M 可训练）输出每 token 的路径/通道调整值（每层 top-k 路由偏置
    # 与通道缩放），50M 参数模型全程冻结。"用 3.5M 训练一个 50M"。
    dual_engine: bool = False      # 开启双引擎模式（优化器只含控制器变量）
    dual_C: int = 224              # 计算器签名/控制器表征维 C
    dual_hidden: int = 320        # 控制器 trunk 隐层（控制可训练预算 <=3.5M）
    dual_fix_gain: float = 0.5    # 固定直通项增益（0 变量）：计算值直接调整通道的初始幅度

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        if self.use_rope:
            self.head_dim = self.d_model // self.n_heads  # 320/8=40
        assert self.K_teacher_slots + self.K_anchor_slots < self.K, \
            "teacher + anchor slots must leave room for task operators"

    @property
    def frozen_params_per_layer(self) -> int:
        """K operators, each d×d + d bias"""
        return self.K * (self.d_model ** 2 + self.d_model)

    @property
    def lora_params_per_layer(self) -> int:
        """K operators × 2 × r × d"""
        if self.lora_r == 0:
            return 0
        return self.K * 2 * self.lora_r * self.d_model

    @property
    def total_params_approx(self) -> int:
        """Rough estimate of total parameters (frozen + trainable)."""
        frozen = self.n_layers * self.frozen_params_per_layer
        lora = self.n_layers * self.lora_params_per_layer
        # Per-layer trainable: Q,K,V,O projections + router + norms
        attn_params = 4 * self.d_router * self.d_model  # Q,K,V,O all d_r×d or d×d_r
        router_params = self.d_router * self.d_model + self.K * self.d_router
        norm_params = 2 * self.d_model
        layer_trainable = attn_params + router_params + norm_params
        # Fix 3: residual escape channel (d*de + de*d + gate)
        escape = 2 * self.d_model * self.d_escape + self.d_model + 17
        layer_trainable += escape if self.use_residual_escape else 0
        embedding = self.vocab_size * self.d_model
        head = 0 if self.tie_weights else self.vocab_size * self.d_model
        final_norm = self.d_model
        trainable = self.n_layers * layer_trainable + embedding + head + final_norm
        total = frozen + lora + trainable
        return total

    def summary(self) -> str:
        frozen = self.n_layers * self.frozen_params_per_layer
        lora = self.n_layers * self.lora_params_per_layer
        esc = self.n_layers * (2 * self.d_model * self.d_escape + 17) if self.use_residual_escape else 0
        return (
            f"FNR {self.label} preset:\n"
            f"  d={self.d_model}, L={self.n_layers}, K={self.K}, k={self.k}, "
            f"d_r={self.d_router}, heads={self.n_heads}\n"
            f"  Vocab: {self.vocab_size}, MaxSeq: {self.max_seq_len}\n"
            f"  Frozen/layer: {self.frozen_params_per_layer/1e6:.2f}M × {self.n_layers} = "
            f"{frozen/1e6:.1f}M\n"
            f"  LoRA(r={self.lora_r})/layer: {self.lora_params_per_layer/1e6:.2f}M × {self.n_layers} = "
            f"{lora/1e6:.2f}M\n"
            f"  Escape(d_e={self.d_escape})/layer: {esc/max(self.n_layers,1)/1e3:.0f}K (fix #3)\n"
            f"  Total ≈ {self.total_params_approx/1e6:.1f}M\n"
            f"  Ceiling fixes: spectral_init={self.use_spectral_init} "
            f"(t={self.K_teacher_slots},a={self.K_anchor_slots}) "
            f"escape={self.use_residual_escape} "
            f"LoRA_tube={self.lora_r} "
            f"lib_growth={self.use_library_growth}\n"
            f"  Universal: shared_library={self.shared_library} shared_router={self.shared_router} "
f"freeze_norms={self.freeze_norms} single_gemm={self.single_gemm} "
f"mid_zero_gate={self.use_mid_zero_gate}\n"
            f"  Other: RoPE, RMSNorm, GELU, TiedHead, ProgressiveLoRA, AdaptiveEntropy, GroupedGEMM"
        )


def get_config() -> FNRConfig:
    cfg = FNRConfig()
    print(cfg.summary())
    return cfg


# ─────────────────────────────────────────────
#  8M "frozen-limit" preset
# ─────────────────────────────────────────────
def get_config_8m() -> FNRConfig:
    """
    8M-parameter FNR that pushes the frozen-to-trainable ratio to its
    practical limit (the "多用固定程序、只训练极少变量" philosophy):

      * The frozen operator bank (K·d²·L) dominates the parameter count.
      * LoRA is OFF (r=0) → no soft extension of the frozen operators;
        coverage comes instead from Fix1 spectral init (frozen, 0 params)
        + Fix3 narrow escape (trainable, ~0.03M total).
      * Router / value / escape dims are shrunk so the only trainable
        parameters are: tied embedding (floor) + tiny attention/router/escape.

    At the 8M scale the tied embedding (vocab·d ≈ 0.8M) is a hard floor on
    the trainable count, so the achievable frozen:trainable ratio is ~7:1
    (vs 17:1 at 100M, where d²·K·L grows faster than the embedding floor).
    """
    cfg = FNRConfig()
    cfg.label = "8M-frozen-limit"
    # shrink d / layers / K so the whole model lands at ~8M params
    cfg.d_model = 96
    cfg.n_layers = 10
    cfg.K = 76
    cfg.k = 4
    cfg.d_router = 24
    cfg.d_v = 24
    cfg.n_heads = 6                       # head_dim = 96/6 = 16
    # freeze the operators to the limit: LoRA OFF
    cfg.lora_r = 0
    cfg.lora_alpha = 1.0
    cfg.lora_progressive = False
    # narrow escape channel = the minimal trainable "universal" core
    cfg.d_escape = 8
    # keep Fix1 (frozen, coverage-constructive) + Fix3 (narrow escape)
    cfg.use_spectral_init = True
    cfg.K_teacher_slots = 4
    cfg.K_anchor_slots = 2
    cfg.c_mix = 0.3
    cfg.use_residual_escape = True
    cfg.use_library_growth = False
    # 隐藏头零门控（除首尾）：中间 8 个 block 的残差增量各乘 1 个 0-init
    # 标量门控 —— 初始态中间层是恒等映射（不扰乱已算出的 token 值），
    # 门控随训练自然打开；新增可训练变量仅 L-2 = 8 个标量（极少变量）。
    cfg.use_mid_zero_gate = True
    # 吞吐追 dense（MX150 实测结论）：
    #   * 低秩因子化 bank_rank=24 实测 SLOWER（5800 vs 933 ms/step）—— 老卡上
    #     把一次 batched GEMM 拆两次小 GEMM+两次 gather 反而更慢，故 bank_rank=0。
    #   * 真正针对"算子子层占 96% 成本"的杠杆是 k=4→k=2：实测 937→622 ms/step
    #     （4.0k→6.7k tok/s），与 dense 差距从 3.7× 缩到 2.4×。k=2 仍保留 top-k
    #     路由语义（只是每 token 路由 2 个算子而非 4 个），"多固定程序少变量"不变。
    cfg.k = 2
    cfg.bank_rank = 0
    cfg.tie_weights = True
    cfg.fp16 = False                     # Pascal MX150: fp32 faster
    # keep the 1024 RoPE table; training windows can be shorter
    cfg.max_seq_len = 1024
    # v2 fastest-profile: B16S256ga1 (peak ~1GB, cooler -> less thermal
    # throttle); 5000 steps = 20.5M tokens (same data as v1, not reduced)
    cfg.warmup_steps = 300
    print(cfg.summary())
    return cfg


# ─────────────────────────────────────────────
#  Universal "fixed program" preset
# ─────────────────────────────────────────────
def get_config_universal() -> FNRConfig:
    """
    8M-scale FNR pushed to the universal fixed-program limit.

    Philosophy: 多固定程序、极少变量 → 把模型变成一个"万能程序"。
      * shared_library : 全 L 层共享 1 份冻结算子库（万能指令集，只存 1 份，
        删掉 L 份重复的算子存储与每步 L× 的 gather/搬运）。
      * shared_router  : 1 份路由控制器（operator_embeds+proj 只 1 份），
        每层只靠一个 0 参数"层模板"偏置调用同一程序的不同段。
      * freeze_norms   : 删掉 L 份重复的单位缩放可训练变量（norm 权重全 1 冻结）。
      * single_gemm    : 提高 W_sel 预算，一次性 GEMM 处理整 batch，
        删掉分块的重复 gather + 多次 kernel 发射。
      * use_mid_zero_gate + escape : 可训练核心只剩"如何路由" + 一个最窄逃逸。
    """
    cfg = FNRConfig()
    cfg.label = "Universal"
    cfg.d_model = 96
    cfg.n_layers = 10
    cfg.K = 76
    cfg.k = 4
    cfg.d_router = 24
    cfg.d_v = 24
    cfg.n_heads = 6
    # freeze the operators to the limit: LoRA OFF
    cfg.lora_r = 0
    cfg.lora_alpha = 1.0
    cfg.lora_progressive = False
    cfg.d_escape = 8
    cfg.use_spectral_init = True
    cfg.K_teacher_slots = 4
    cfg.K_anchor_slots = 2
    cfg.c_mix = 0.3
    cfg.use_residual_escape = True
    cfg.use_library_growth = False
    # 万能固定程序开关
    cfg.shared_library = True
    cfg.shared_router = True
    cfg.shared_K = 200
    cfg.freeze_norms = True
    cfg.single_gemm = True
    cfg.use_mid_zero_gate = True
    cfg.tie_weights = True
    cfg.fp16 = False
    cfg.max_seq_len = 1024
    cfg.warmup_steps = 300
    print(cfg.summary())
    return cfg


# ─────────────────────────────────────────────────────────────
#  Quality-oriented universal preset (语言质量优化：用最少变量换质量)
# ─────────────────────────────────────────────────────────────
def get_config_quality() -> FNRConfig:
    """
    与 universal 相同的「万能固定程序」底座，但为了把语言质量
    (流畅度/困惑度) 提起来，在仍然保持「少量可训练变量」的前提下
    做三件事：
      * shared_K 200 -> 300：共享冻结库扩容到 300 槽位，扩大冻结可达集
        E_{K,k} 的谱覆盖，不增加任何可训练变量（只加存储型冻结变量）。
      * d_escape 8 -> 16：逃逸通道加宽一倍，可训练核 0.92M -> ~1.4M，
        仍远少于同格式 TinyTransformer 的 1.6M（保留冻结红利）。
      * tau_start 5->3, tau_end 0.5->0.7, lambda_entropy 0.05->0.08：
        路由更平滑 + 熵正则稍强，减少「单一算子独吞」造成的死记
        和 greedy-EOS 行为。
    数据侧（配合使用）：质量问题根源是欠训练（20.5M << Chinchilla 甜区
    160M），本预设搭配 --max_steps 15000 @ B16 S256（≈61M token）
    才能看到质量提升；如果只能训 20.5M，quality 版相对 universal 版
    在困惑度上只会改善边际，不会质变。
    """
    cfg = get_config_universal()
    cfg.label = "Quality-universal"
    cfg.shared_K = 300
    cfg.d_escape = 16
    cfg.tau_start = 3.0
    cfg.tau_end = 0.7
    cfg.lambda_entropy = 0.08
    print(cfg.summary())
    return cfg


# ─────────────────────────────────────────────────────────────
#  双引擎 50M 预设：3.5M 控制器 + 全冻结 50M（"用 3.5M 训 50M"）
# ─────────────────────────────────────────────────────────────
def get_config_50m() -> FNRConfig:
    """
    双引擎预训练（本轮"不实测"落地为代码 + 报告）：
      冻结大引擎 = 独立冻结库 50M（d160/L12/K156 独立 bank；
        冻结算子库 12*156*(160^2+160) ≈ 45.5M + 嵌入/注意力/路由/norm/头
        其余部分全部冻结 = 全 50M 冻结，冻结算子占比 91%）。
      可训练小引擎 = PathController ≈3.2M（<=3.5M 预算）：
        token 值表（用 FunctionCalculator 输出初始化）+ trunk
        + 零初始化 path/chan 头 -> 输出"遇到该 token 时所有通道
        应改变的路径连接权重"（每层 top-k 路由偏置 + 通道缩放）。
    优化器只更新控制器变量；50M 冻结部分全程 0 梯度。
    用法：python train.py --preset 50m --max_steps 5000 --batch_size 8 --seq_len 256
    """
    cfg = FNRConfig()
    cfg.label = "50M-dual-engine"
    cfg.d_model = 160
    cfg.n_layers = 12
    cfg.K = 156
    cfg.k = 2                        # 吞吐追 dense（MX150 实测 k4=2729ms -> k2=1659ms）
    cfg.d_router = 40
    cfg.d_v = 40
    cfg.n_heads = 8                       # head_dim = 160/8 = 20
    cfg.lora_r = 0
    cfg.lora_progressive = False
    cfg.d_escape = 0                      # 双引擎下逃逸通道关闭（质量由控制器承担）
    cfg.use_residual_escape = False
    cfg.use_spectral_init = True
    cfg.K_teacher_slots = 4
    cfg.K_anchor_slots = 2
    cfg.c_mix = 0.3
    cfg.use_library_growth = False
    # 冻结大引擎：嵌入/注意力/路由/norm/头 全部冻结（见 FNRModel dual_engine 分支）
    cfg.dual_engine = True
    cfg.dual_C = 224
    cfg.dual_hidden = 320
    cfg.dual_fix_gain = 0.5
    # 显存：50m 靠 mb_budget=96 的 W_sel 分块（live gather 封顶 ~96MB）保证
    # B8=927MB / B16=1563MB 均在 2GB 卡内。实测 fp16 在 Pascal 不提速且会产生
    # 双份 bank（fp32 master + fp16 cache），故 50m 保持 fp32 单份（不省反增），
    # frozen_fp16 保持默认 False。
    # 万能程序化 0 变量机制与双引擎正交：50M 用独立 bank（共享开关关闭）
    cfg.use_mid_zero_gate = False
    cfg.tie_weights = True
    cfg.fp16 = False
    cfg.max_seq_len = 1024
    cfg.warmup_steps = 300
    print(cfg.summary())
    return cfg
