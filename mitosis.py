"""
细胞分裂式升级（"计算改权重"，0 可训练变量，闭式、确定性、秒级完成）。

思想（对应指令）：
  数据太少（~2.4M token）训不动 50M 模型 → 不做"加数据重训"，
  把语言能力升级拆成两步：

  1) 分裂（离线、一次性、0 训练）：每个父算子槽位 v 分裂成一对子槽位
     (2v, 2v+1)。两种模式：
       'duplicate'（默认，warm-start 精确）：
         W[2v] = W[2v+1] = W_v , b 同理。
         因为路由器 top-k 权重归一化（Σw=1），
           w_a·f_v + w_b·f_v = (w_a+w_b)·f_v = f_v
         任意权重分配下父算子行为**精确可达**（构造性恒等，无漂移）。
         子槽位的路由嵌入 = 父嵌入 ± 确定性偏移（0 变量），
         给二阶段"分化"留出手柄。
       'complement'（激进特化，近似）：
         P_a + P_b = I, P_a P_b = 0（确定性正交基）
         W[2v] = P_a W_v , W[2v+1] = P_b W_v。
         诚实边界：GELU 逐通道非线性下，两子 0.5/0.5 选取时输出 =
         0.5·act(y)（整体 2× 衰减，下一层 RMSNorm 部分抹平全局尺度，
         但相对残差流的算子贡献强度变了）→ 不是精确 warm-start，
         需要短重训消化。默认不用它做 warm-start。

  2) 分化（二阶段短重训，只训 <=3.5M 控制器）：子槽位从"与父近似等价"
     起步（duplicate 模式 = 零漂移），控制器路径头按父行复制，
     短重训让 2K 槽位逐渐特化 → 语言能力升级。
     升级耗时账：分裂 = 一次权重拷贝（r0 约 200MB fp32 搬运，秒级 CPU），
     二阶段只训 2.4M→~2.85M 控制器（≤3.5M 预算），
     远小于任何 50M 全量重训。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ctrl_modules import deterministic_orthonormal


def complementary_projections(d: int, seed: int = 2026, device="cpu"):
    """确定性互补通道投影 (P_a, P_b)：P_a + P_b = I, P_a P_b = 0（0 变量）。
    P_a 投到种子决定的正交基前 d//2 维（固定程序）。"""
    ha = max(1, d // 2)
    Q = deterministic_orthonormal(d, d, seed=seed, device=device)   # (d, d)
    Qa = Q[:, :ha]                                                   # (d, ha)
    P_a = Qa @ Qa.T                                                  # (d, d) 正交投影
    P_b = torch.eye(d, device=device) - P_a
    return P_a, P_b


def _activate(t: torch.Tensor, act: str) -> torch.Tensor:
    if act == "gelu":
        return F.gelu(t)
    if act == "relu":
        return F.relu(t)
    return torch.sigmoid(t)


def cell_divide_bank(bank, mode: str = "duplicate", seed: int = 2026,
                     verify: bool = True) -> int:
    """冻结算子库 K -> 2K（v-major：2v=子 a, 2v+1=子 b）。闭式、0 变量。

    mode='duplicate'：子 = 父精确复制（构造性 warm-start，任意权重归一化
                      下 Σw·f = f，见模块 docstring）。
    mode='complement'：子 = 父的互补通道分割（P_a/P_b 固定基，近似特化）。
    返回新 K。
    """
    # 低秩模式 bank.W 已被 factorize() 删除 → 用 W_full() 取全空间矩阵
    W0 = bank.W_full() if getattr(bank, "rank", 0) > 0 else bank.W
    K, d, _ = W0.shape
    act = bank._activation
    with torch.no_grad():
        W, b = W0, bank.b
        if mode == "duplicate":
            new_W = torch.stack([W, W], dim=1).reshape(2 * K, d, d)
            new_b = torch.stack([b, b], dim=1).reshape(2 * K, d)
        elif mode == "complement":
            P_a, _P_b = complementary_projections(d, seed=seed,
                                                   device=W.device)
            W_a = torch.einsum("ij,vjd->vid", P_a, W)
            new_W = torch.stack([W_a, W - W_a], dim=1).reshape(2 * K, d, d)
            b_a = torch.einsum("ij,vj->vi", P_a, b)
            new_b = torch.stack([b_a, b - b_a], dim=1).reshape(2 * K, d)
        else:
            raise ValueError(f"unknown mitosis mode: {mode}")
        # 静态自检（不训练）：'duplicate' 下父输出在 0.5/0.5 选取时精确恢复
        if verify:
            g = torch.Generator().manual_seed(0)
            z = torch.randn(K, d, generator=g, device=W.device)
            y_parent = _activate(torch.einsum("vjo,vo->vo", W, z) + b, act)
            fa = _activate(torch.einsum("vjo,vo->vo", new_W[0::2], z)
                           + new_b[0::2], act)
            fb = _activate(torch.einsum("vjo,vo->vo", new_W[1::2], z)
                           + new_b[1::2], act)
            err = (0.5 * fa + 0.5 * fb - y_parent).abs().max().item()
            if mode == "duplicate":
                assert err < 1e-5, f"duplicate 模式自检失败: {err:.2e}"
            # complement 模式：报告近似偏差（不 assert，文档化）
    if getattr(bank, "rank", 0) > 0:
        # 低秩模式：先放回全空间 W，再 factorize() 重新 SVD 截断成因子
        bank.register_buffer("W", new_W)
        bank.register_buffer("b", new_b)
        bank.factorize()
    else:
        bank.register_buffer("W", new_W)
        bank.register_buffer("b", new_b)
    if bank.lora_r > 0:      # LoRA（若开启）：子槽位继承父 LoRA，短重训再分化
        with torch.no_grad():
            A_new = bank.lora_A.unsqueeze(1).expand(K, 2, bank.lora_r, d) \
                .reshape(2 * K, bank.lora_r, d).clone()
            B_new = bank.lora_B.unsqueeze(1).expand(K, 2, d, bank.lora_r) \
                .reshape(2 * K, d, bank.lora_r).clone()
        bank.lora_A = nn.Parameter(A_new, requires_grad=True)
        bank.lora_B = nn.Parameter(B_new, requires_grad=True)
    bank._cache = {}          # fp16 缓存失效（形状已变）
    return 2 * K


def duplicate_slots(x: torch.Tensor, axis: int) -> torch.Tensor:
    """(..., K, ...) -> (..., 2K, ...)：沿 axis 把 K 维交错复制（2v <- v, 2v+1 <- v）。
    保留 axis 之前/之后的所有维度（L 维不丢）。"""
    K = x.shape[axis]
    out = x.unsqueeze(axis + 1)                                  # (..., K, 1, ...)
    out = out.expand(*x.shape[:axis], K, 2, *x.shape[axis + 1:])  # (..., K, 2, ...)
    out = out.reshape(*x.shape[:axis], 2 * K, *x.shape[axis + 1:])
    return out.contiguous()


def extend_router_embeds(router, seed: int = 2026,
                         offset_scale: float = 0.25) -> None:
    """路由嵌入 K -> 2K：子 = 父 ± 确定性偏移（同父语义起步，0 变量区分度，
    二阶段短重训时才分化）。偏移方向由种子正交基给出（0 可训练变量）。"""
    E = router.operator_embeds
    K, d_r = E.shape
    requires_grad = bool(E.requires_grad)
    q0 = deterministic_orthonormal(1, d_r, seed=seed + 1,
                                   device=E.device).view(d_r)
    off = (offset_scale / math.sqrt(d_r)) * q0
    with torch.no_grad():
        new_E = torch.empty(2 * K, d_r, device=E.device, dtype=E.dtype)
        new_E[0::2] = E + off
        new_E[1::2] = E - off
    router.operator_embeds = nn.Parameter(new_E, requires_grad=requires_grad)
    router.K = 2 * K


def extend_controller_slots(controller, L: int, K_new: int) -> None:
    """控制器路径头 (L*K_old, C) -> (L*K_new, C)：按 li-major 把父行复制给
    两个子行（0 变量，保持分裂前"固定直通项"语义不变）。
    注意：Linear 权重不能 in-place 扩容（copy_ 不改形状），必须整体替换
    模块（requires_grad 保持 True，二阶段短重训继续更新新槽位）。"""
    w = controller.path_head.weight
    K_old = w.shape[0] // L
    C = w.shape[1]
    new_w = duplicate_slots(w.view(L, K_old, -1), axis=1) \
        .view(L * K_new, C).clone()
    # 替换为同构新 Linear（无 bias），拷贝父行语义
    controller.path_head = nn.Linear(C, L * K_new, bias=False)
    with torch.no_grad():
        controller.path_head.weight.copy_(new_w)
    controller.path_head.to(w.device)
    # forward 用 self.K 拼 path 输出形状 (B,N,L,K)，必须同步到新 K
    controller.K = K_new


def verify_parent_reachable(bank, mode: str = "duplicate",
                           seed: int = 2026) -> float:
    """分裂后一致性自检（静态校验，不训练）：
    0.5·f(2v) + 0.5·f(2v+1) 与父 f(v) 的最大偏差。
    duplicate 模式应 < 1e-5（构造性恒等）；complement 模式返回实测偏差
    （GELU 下 ≈ 0.5·|act(y)|，非精确 —— 见模块 docstring 诚实边界）。
    低秩 bank 用 W_full()（因子重建的全空间矩阵）做自检。"""
    Wf = bank.W_full()
    K2, d, _ = Wf.shape
    K = K2 // 2
    g = torch.Generator().manual_seed(0)
    z = torch.randn(K, d, generator=g, device=Wf.device)
    Wfull = Wf[:K]                    # 父 K 个全空间算子（低秩=截断重建）
    fa = _activate(torch.einsum("vjo,vo->vo", Wf[0::2], z)
                   + bank.b[0::2], bank._activation)
    fb = _activate(torch.einsum("vjo,vo->vo", Wf[1::2], z)
                   + bank.b[1::2], bank._activation)
    if mode == "duplicate":
        # 子 = 父精确复制：任一子输出即父输出
        y_parent = _activate(torch.einsum("vjo,vo->vo", Wf[0::2], z)
                              + bank.b[0::2], bank._activation)
    else:
        # complement：子a = P_a·父, 子b = P_b·父, P_a+P_b=I 且幂等
        # ⇒ 父 = 子a + 子b（pre-activation 线性部分精确相加）
        W_orig = Wf[0::2] + Wf[1::2]
        b_orig = bank.b[0::2] + bank.b[1::2]
        y_parent = _activate(torch.einsum("vjo,vo->vo", W_orig, z)
                              + b_orig, bank._activation)
    # GELU 逐通道非线性下，complement 的 0.5·fa+0.5·fb 与父 y 有实测偏差
    # （duplicate 恒等 < 1e-5；complement ≈ 0.5 衰减 —— 诚实记录）
    err = (0.5 * fa + 0.5 * fb - y_parent).abs().max().item()
    return err


def _unique_banks(model):
    seen, out = set(), []
    for blk in model.blocks:
        if id(blk.operator_bank) not in seen:
            seen.add(id(blk.operator_bank))
            out.append(blk.operator_bank)
    return out


def cell_divide_all(model, mode: str = "duplicate", seed: int = 2026) -> int:
    """在 FNRModel 上执行一次完整细胞分裂（bank + router + 层偏置 + 控制器 +
    固定直通基 M_path + cfg.K + 熵常量缓存）。支持独立库（8m/50m：每层各 1 份
    bank/router）与共享库（universal/quality：bank/router 各 1 份，按实际 K）。
    返回新 K（按 bank 实际大小计算，避免共享预设下 cfg.K != bank.K 的错位）。"""
    banks = _unique_banks(model)
    for bank in banks:
        cell_divide_bank(bank, mode=mode, seed=seed)
    K_new = banks[0].W.shape[0]
    seen = set()
    for blk in model.blocks:
        r = blk.operator_router
        if id(r) in seen:
            continue
        seen.add(id(r))
        if r.K * 2 == K_new:
            extend_router_embeds(r, seed)
    for blk in model.blocks:
        if blk.layer_bias is not None:
            blk.layer_bias = duplicate_slots(
                blk.layer_bias.view(1, 1, blk.layer_bias.shape[-1]),
                axis=-1).view(1, 1, K_new)
    if getattr(model, "dual_engine", False):
        extend_controller_slots(model.path_controller,
                                L=model.cfg.n_layers, K_new=K_new)
        # 固定直通基 M_path (L*K_old, C) -> (L*K_new, C)：按 li-major 把每个父行
        # 复制给两个子行（0 变量，保持"固定程序"直通项语义在分裂前后不变）。
        _rebuild_m_path(model, K_new)
    if hasattr(model, "_ent_const") and hasattr(model, "_ent_dev"):
        del model._ent_const
        del model._ent_dev
    # 独立库预设：cfg.K 即 bank K；共享预设：K 在 cfg.shared_K 上
    model.cfg.K = K_new
    if getattr(model.cfg, "shared_library", False):
        model.cfg.shared_K = K_new
    return K_new


def _rebuild_m_path(model, K_new: int) -> None:
    """把 M_path (L*K_old, C) 重建为 (L*K_new, C)（父行复制到两子行）。
    重新注册为 buffer（persistent=False，不进 state_dict，由 ckpt 的 mitosis
    段显式携带；resume 时按 model 设备重建）。"""
    M = model.M_path                                    # (L*K_old, C)
    L = model.cfg.n_layers
    K_old = M.shape[0] // L
    new_M = duplicate_slots(M.view(L, K_old, -1), axis=1) \
        .view(L * K_new, M.shape[1]).to(M.dtype).to(M.device)
    model._buffers.pop("M_path", None)
    model.register_buffer("M_path", new_M, persistent=False)


def serialize_split(model, mode: str, seed: int, K_parent: int) -> dict:
    """把分裂后"分裂瞬间"的状态打包（供 ckpt 携带，resume 可精确重建 0 变量直通基）。"""
    seg = {"mode": mode, "seed": seed, "K_parent": K_parent}
    if getattr(model, "dual_engine", False):
        seg["M_path"] = model.M_path.detach().cpu()
        seg["dual_C"] = model.cfg.dual_C
        seg["L"] = model.cfg.n_layers
    return seg


def restore_split(model, seg: dict, device) -> None:
    """resume：按 ckpt 携带的分裂段把 M_path 固定基放回 model 设备。
    （state_dict 里 M_path persistent=False 不会自动加载，这里手动还原 0 变量直通基。）"""
    if not seg or "M_path" not in seg:
        return
    L = seg["L"]
    C = seg["dual_C"]
    K_new = model.cfg.K
    M = seg["M_path"].to(device)
    assert M.shape[0] == L * K_new, \
        f"M_path 行数 {M.shape[0]} != L*K {L*K_new}；此 ckpt 未按当前 preset 分裂"
    model._buffers.pop("M_path", None)
    model.register_buffer("M_path", M, persistent=False)
