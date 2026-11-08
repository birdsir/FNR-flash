"""
FNR-100M 正式训练脚本（中文对话模型）
真实数据: corpus/train_tokens.pt + val_tokens.pt + tokenizer.json
显存适配: 分块 GEMM, fp16 autocast, B=4/S=128 (peak ~1.9GB on MX150 2GB)
特性: 文件日志 + 定期 val loss + 定期生成抽检 + 定期 ckpt + final.pt
用法:
  python train.py                      # 默认 B=4 S=128, 10000 步
  python train.py --max_steps 20000 --batch_size 4 --seq_len 128
"""
import argparse, os, sys, math, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (FNRConfig, get_config, get_config_8m,
                    get_config_universal, get_config_quality, get_config_50m)
from fnr_model import FNRModel
from tokenizer import BPE

BASE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(BASE, "train.log")


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


class PackedDataset(Dataset):
    """Random windows over a packed 1-D token stream.
    Auto-detects int16-packed streams: load once as int16, cast per-window
    to long on the fly — halves CPU RAM and load bandwidth for the 1-D
    token stream (vocab 8192 < 32768 fits int16).
    order: 可选段序（L3 课程，dedup_filter.py 输出的 easy->hard 段号数组）。
      提供时按段序读段（不随机窗口），使"熟悉数据在前、新颖数据在后"。"""
    def __init__(self, data_path, seq_len, order=None):
        raw = torch.load(data_path, map_location="cpu", weights_only=True)
        if raw.dtype == torch.int16:
            self.data = raw
            self._need_cast = True
        elif raw.dtype == torch.int32:
            self.data = raw
            self._need_cast = True
        else:
            self.data = raw.long()
            self._need_cast = False
        self.seq_len = seq_len
        nseg = max(1, self.data.numel() // seq_len)
        if order is not None:
            order = [int(i) % nseg for i in order]
            self.order = order
            self.n = len(order)
        else:
            self.order = None
            self.n = nseg

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        seg = (self.order[idx] if self.order is not None else
               (idx % self.n))
        start = seg * self.seq_len
        end = start + self.seq_len
        if end > self.data.numel():
            end = self.data.numel(); start = end - self.seq_len
        w = self.data[start:end]
        if self._need_cast:
            w = w.long()
        return w.contiguous()


class Trainer:
    def __init__(self, cfg, model, args):
        self.cfg, self.model, self.args = cfg, model, args
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=cfg.lr,
                                            weight_decay=cfg.weight_decay,
                                            betas=(0.9, 0.95))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda s: self._cosine(s, cfg.warmup_steps, cfg.max_steps))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = cfg.fp16 and self.device.type == "cuda"
        self.model.to(self.device)
        self.scaler = torch.amp.GradScaler("cuda") if self.use_fp16 else None
        self.ckpt_dir = Path(args.output)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        # （KD 钩子保留但方案不用：无 teacher 时全路径等价纯 CE）
        self.teacher = getattr(args, "teacher", None)
        self.kd_alpha = float(getattr(args, "kd_alpha", 0.0))
        self.kd_temp = float(getattr(args, "kd_temp", 2.0))
        # L3 EMA：只对可训练核（router/escape/embed/controller）平滑
        self.ema_decay = float(getattr(args, "ema_decay", 0.0))
        self.ema = None
        if self.ema_decay > 0:
            self.ema = {n: p.detach().clone()
                        for n, p in self.model.named_parameters()
                        if p.requires_grad}

    @staticmethod
    def _cosine(step, warmup, max_steps):
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, max_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    def train_step(self, batch, step, accumulate=False):
        """accumulate=True: backward only (keep grads, no step).
        accumulate=False: backward + clip + optimizer.step + scheduler.step.
        NOTE: caller is responsible for zero_grad at the start of the window."""
        self.model.train()
        # non_blocking H2D: overlaps the transfer with the previous step's work
        # (the loader already pins memory when CUDA is available).
        nb = self.device.type == "cuda"
        x = batch if batch.device.type == self.device.type             else batch.to(self.device, non_blocking=nb)
        labels = x.clone()
        kd_loss = None
        if self.scaler:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = self.model(x, labels=labels, training=True, step=step)
                if self.teacher is not None and self.kd_alpha > 0:
                    kd_loss = self._kd_loss(out.logits, self.teacher(x)[0], labels)
                target = kd_loss if kd_loss is not None else out.loss
            self.scaler.scale(target).backward()
        else:
            out = self.model(x, labels=labels, training=True, step=step)
            if self.teacher is not None and self.kd_alpha > 0:
                # L2 KD：CE + alpha·KL(teacher||student)（软目标 T 次缩放）
                tl, _ = self.teacher(x)
                kd_loss = self._kd_loss(out.logits, tl, labels)
                kd_loss.backward()
            else:
                out.loss.backward()
        if not accumulate:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_([p for p in self.model.parameters()
                                     if p.requires_grad], self.cfg.grad_clip)
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()
            # L3 EMA：只更新可训练核的 shadow 权重（数据受限时比单点更稳）
            if self.ema is not None:
                for n, p in self.model.named_parameters():
                    if p.requires_grad and n in self.ema:
                        self.ema[n].mul_(self.ema_decay).add_(
                            p.detach(), alpha=1 - self.ema_decay)
        _loss_val = kd_loss.item() if kd_loss is not None else out.loss.item()
        return {"loss": _loss_val,
                "lr": self.scheduler.get_last_lr()[0],
                "ent": out.routing_entropy.item()}

    def _kd_loss(self, s_logits, t_logits, labels):
        """CE + alpha·KL(softmax(t/T)||softmax(s/T))·T²（Hinton 2015 蒸馏）。
        输入为完整 (B,N,V) logits；内部做 1-token 移位与 reshape。"""
        T = self.kd_temp
        sl = s_logits[:, :-1].reshape(-1, s_logits.size(-1)).float()
        tl = t_logits[:, :-1].reshape(-1, t_logits.size(-1)).float()
        lab = labels[:, 1:].reshape(-1)
        log_s = F.log_softmax(sl / T, dim=-1)
        p_t = F.softmax(tl / T, dim=-1)
        kd = F.kl_div(log_s, p_t, reduction="batchmean") * T * T
        ce = F.cross_entropy(s_logits[:, :-1].reshape(-1, s_logits.size(-1)),
                             lab)
        return (1 - self.kd_alpha) * ce + self.kd_alpha * kd

    def val_loss(self, loader, cap=8):
        self.model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for i, b in enumerate(loader):
                if i >= cap:
                    break
                x = b.to(self.device)
                if self.scaler:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        out = self.model(x, labels=x.clone(), training=False, step=0)
                else:
                    out = self.model(x, labels=x.clone(), training=False, step=0)
                total += out.loss.item()
                n += 1
        return total / max(n, 1)

    def sample_lines(self, tok, n=5):
        self.model.eval()
        seeds = ["用户：你好\n助手：", "今天天气", "人工智能", "《红楼梦》", "学习",
                 "用户：失眠怎么办\n助手：", "春天到了", "中国的首都"]
        seeds = seeds[:n]
        lines = []
        with torch.no_grad():
            for s in seeds:
                ids = tok.encode(s, add_bos=True)
                x = torch.tensor([ids], dtype=torch.long, device=self.device)
                out = self.model.generate(x, max_new_tokens=32, temperature=0.9,
                                          top_k=50, do_sample=True)
                g = tok.decode(out[0, x.shape[1]:])
                lines.append(f"  {s!r} -> {g[:60]!r}")
        return "\n".join(lines)

    def save_ckpt(self, step, val=None, final=False):
        sd = self.model.state_dict()
        if final and self.ema is not None:
            # L3：最终 ckpt 用 EMA shadow 权重（可训练核部分更平滑）
            for n, p in self.model.named_parameters():
                if p.requires_grad and n in self.ema:
                    sd[n] = self.ema[n].detach()
        d = {"step": step, "val_loss": val,
             "model_state_dict": sd,
             "config": self.cfg.__dict__}
        name = "final.pt" if final else f"step_{step}.pt"
        p = self.ckpt_dir / name
        torch.save(d, p)
        return str(p)


class DualStreamPipeline:
    """CPU/GPU 双流双缓冲流水线（数据不足 2h 方案的吞吐放大器）。

    强利用两端：
      * CPU 端：DataLoader num_workers 并行切窗/pin/类型转换（不阻塞主核）；
      * GPU 端：两个 CUDA stream —— s_copy 异步 H2D（pinned 双缓冲），
        s_comp 跑前反向。step n 的 GEMM 与 step n+1 的 H2D **重叠**，
        把 ~2-5ms 的 H2D 延迟藏进 2.7s 的 GEMM 步里（发射间隙归零）。
    无 CUDA 时退化为直通（不报错，仅失去重叠收益）。"""
    def __init__(self, device):
        self.dev = torch.device(device)
        self.ok = self.dev.type == "cuda" and torch.cuda.is_available()
        if self.ok:
            self.s_copy = torch.cuda.Stream()
            self.s_comp = torch.cuda.Stream()
            self.buf = [torch.empty(0, dtype=torch.int64, device=self.dev) for _ in range(2)]
            self.ev = [None, None]
            self.slot = 0

    def prefetch(self, host_batch):
        """在 s_copy 流上把 host（pinned）batch 搬到 GPU（后台，与当前步 GEMM 重叠）。"""
        if not self.ok:
            return
        s = self.slot
        if hasattr(host_batch, "is_pinned") and host_batch.is_pinned():
            src_t = host_batch
        else:
            src_t = host_batch.pin_memory()
        with torch.cuda.stream(self.s_copy):
            self.buf[s] = src_t.to(self.dev, non_blocking=True)
        self.ev[s] = self.s_copy.record_event()
        self.slot ^= 1

    def get(self):
        """取上一发 H2D 完成的 GPU batch（本流上等 s_copy 事件，~微秒级；
        H2D 已在上一轮 GEMM 期间后台完成，通常事件早已 ready）。"""
        if not self.ok:
            return None
        s = self.slot ^ 1
        if self.ev[s] is not None:
            self.ev[s].synchronize()
            self.ev[s] = None
        return self.buf[s]


class SpeedPhasePicker:
    """三阶段自适应 grad_accum 调度（"暴力 → 优化 → 暴力多数据"）。

    不重建 DataLoader，只通过每步实际采用的 `acc`（有效 batch = B*acc）切档：
      * phase1 暴力（前 ~warmup+500 步）：acc = acc_max（吃更多数据/步，
        学习率最高，快速降 loss）——一开始就"暴力"。
      * phase2 优化（速度"下来"时）：每 50 步用滚动平均 ms/step 与基线比，
        若热降频/卡顿使速度掉 >15%，acc 降到 1（固定档，少算、保速度）——
        "等速度下来再进行优化训练"。
      * phase3 暴力多数据（速度"上来"时）：速度恢复（连续 3 个窗口 ms/step
        不高于基线 5% 以内），acc 提到 acc_max（每步多吞吐数据）——
        "等速度上来再暴力处理多数据"。

    返回 (acc, phase_name)。phase_name ∈ {bruteforce, optimize, moredata}。
    """
    def __init__(self, acc_max: int, base_ms: float,
                 degrade_th: float = 0.15, recover_th: float = 0.05,
                 warmup: int = 200, window: int = 50):
        self.acc_max = max(1, acc_max)
        self.base_ms = base_ms
        self.degrade_th = degrade_th
        self.recover_th = recover_th
        self.warmup = warmup
        self.window = window
        self._ms_hist = []          # 最近每步 ms/step（窗口）
        self._pre_hist = []         # 暴力预热期的全部 ms（用来定基线）
        self._step_count = 0
        self._recover_streak = 0
        self._phase = "bruteforce"
        # 基线懒初始化：若 base_ms 为 None，取暴力预热期的滚动中位数
        if self.base_ms is None:
            self._base_ready = False
        else:
            self._base_ready = True

    def record(self, ms_per_step: float, gstep: int):
        """记录一步耗时并按阶段规则更新相位。返回 (acc_for_next, phase_name)。"""
        if gstep < self.warmup:
            # 暴力期：acc 恒为 max（一开始就暴力）
            self._pre_hist.append(ms_per_step)
            if len(self._pre_hist) >= 20 and not self._base_ready:
                # 懒定基线 = 暴力期中位数（避免首步冷启动抖动）
                self.base_ms = sorted(self._pre_hist)[len(self._pre_hist) // 2]
                self._base_ready = True
            self._phase = "bruteforce"
            return self.acc_max, "bruteforce"

        self._ms_hist.append(ms_per_step)
        if len(self._ms_hist) > self.window:
            self._ms_hist.pop(0)
        # 若基线仍未定（例如 warmup 内 <20 步就跳变），用已有数据兜底
        if not self._base_ready:
            self.base_ms = max(1e-6, sum(self._pre_hist or [ms_per_step]) /
                              max(1, len(self._pre_hist)))
            self._base_ready = True
        # 用滚动窗口平均速度判断"下来 / 上来"（至少攒 window//2 个样本再判）
        if len(self._ms_hist) < self.window // 2:
            return self.acc_for(), self._phase
        recent = sum(self._ms_hist) / len(self._ms_hist)
        if self._phase == "bruteforce":
            # 过了暴力预热期，默认进入"优化"档（省算、护速度）
            self._phase = "optimize"
        if self._phase == "optimize":
            if recent <= self.base_ms * (1.0 + self.recover_th):
                # 速度"上来"：连续 3 个窗口恢复到基线 5% 以内 → 进 moredata（acc_max）
                self._recover_streak += 1
                if self._recover_streak >= 3:
                    self._phase = "moredata"
                    self._recover_streak = 0
            else:
                self._recover_streak = 0
        if self._phase == "moredata":
            if recent > self.base_ms * (1.0 + self.degrade_th):
                # 多数据暴力中速度再次"下来"（慢于基线 15%）→ 退回 optimize
                self._phase = "optimize"
                self._recover_streak = 0
        return self.acc_for(), self._phase

    def acc_for(self) -> int:
        if self._phase == "optimize":
            return 1
        return self.acc_max        # bruteforce / moredata 都暴力

    def current(self):
        return self._phase


def choose_dtype(model, cfg, device, iters=6):
    """Auto-bench fp32 vs fp16-autocast on THIS GPU and pick the faster.
    Pascal cards (GP10x, no Tensor Core) are typically faster in fp32;
    Ampere+ with Tensor Core is faster in fp16. Set cfg.fp16 accordingly."""
    if device.type != "cuda":
        return False
    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len),
                      device=device)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]

    def one_step(use_fp16):
        opt = torch.optim.AdamW(trainable, lr=cfg.lr)
        scaler = torch.amp.GradScaler("cuda") if use_fp16 else None
        for _ in range(3):   # warmup
            opt.zero_grad()
            if use_fp16:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    o = model(x, labels=x.clone(), training=True, step=100)
                scaler.scale(o.loss).backward()
                nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                scaler.step(opt); scaler.update()
            else:
                o = model(x, labels=x.clone(), training=True, step=100)
                o.loss.backward()
                nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                opt.step()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            opt.zero_grad()
            if use_fp16:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    o = model(x, labels=x.clone(), training=True, step=200)
                scaler.scale(o.loss).backward()
                nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                scaler.step(opt); scaler.update()
            else:
                o = model(x, labels=x.clone(), training=True, step=200)
                o.loss.backward()
                nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                opt.step()
        torch.cuda.synchronize()
        return (time.time() - t0) / iters

    t32 = one_step(use_fp16=False)
    t16 = one_step(use_fp16=True)
    log(f"[autodtype] fp32={t32*1000:.0f}ms fp16={t16*1000:.0f}ms -> "
        f"{'fp32' if t32 < t16 else 'fp16'} ({abs(t32-t16)/max(t32,t16)*100:.0f}% faster)")
    return bool(t16 < t32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(BASE, "corpus"))
    ap.add_argument("--output", default=os.path.join(BASE, "checkpoints"))
    ap.add_argument("--preset", default="universal",
                    choices=["universal", "quality", "50m", "8m", "100m"],
                    help="model preset: universal=万能固定程序极限, "
                         "quality=universal+扩库/加宽逃逸(语言质量优化), "
                         "50m=双引擎(3.5M控制器+全冻结50M,用3.5M训50M), "
                         "8m=frozen-limit 8M, 100m=original ~100M config")
    ap.add_argument("--max_steps", type=int, default=5000,
                    help="v2: 5000 steps @ B16S256 = 20.5M tokens (same data as v1)")
    ap.add_argument("--batch_size", type=int, default=16,
                    help="v2 MX150 sweet spot (peak ~1GB, least thermal throttle)")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=10000,
                    help="save a model file every N steps (default 10000)")
    ap.add_argument("--grad_accum", type=int, default=1,
                    help="固定 grad_accum（--no_phase 时用）")
    ap.add_argument("--grad_accum_max", type=int, default=4,
                    help="三阶段自适应的'暴力'档最大 grad_accum（吃更多数据/步）")
    ap.add_argument("--no_phase", action="store_true",
                    help="关闭三阶段自适应，用固定 --grad_accum（默认开启：暴力→优化→暴力多数据）")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "fp32"],
                    help="fp16/fp32 to force; auto benches both on this GPU. "
                         "MX150 (no Tensor Core) auto-picks fp32.")
    ap.add_argument("--resume", default=None)
    # ── L3 数据效率（不靠加数据）──
    ap.add_argument("--epochs", type=int, default=1,
                    help="重复数据池次数（数据不足时 2~4 倍复用）")
    ap.add_argument("--ema_decay", type=float, default=0.0,
                    help=">0 时开 EMA（只平滑可训练核），推荐 0.999")
    ap.add_argument("--curriculum", default=None,
                    help="dedup_filter.py 的 curriculum.json 路径（easy->hard 段序）")
    # ── 流水线（CPU/GPU 双流 + 多 worker，2h 时间预算的核心手段）──
    ap.add_argument("--num_workers", type=int, default=0,
                    help="DataLoader CPU worker 数（>=2 时自动开双缓冲流水线）")
    ap.add_argument("--no_pipeline", action="store_true",
                    help="强制关闭 CPU/GPU 双流（对照用）")
    args = ap.parse_args()

    # config (preset selects the model scale)
    if args.preset == "universal":
        cfg = get_config_universal()
    elif args.preset == "quality":
        cfg = get_config_quality()
    elif args.preset == "50m":
        cfg = get_config_50m()
    elif args.preset == "8m":
        cfg = get_config_8m()
    else:
        cfg = get_config()
    cfg.max_steps = args.max_steps
    cfg.batch_size = args.batch_size
    cfg.max_seq_len = args.seq_len   # RoPE table is built to 1024; windows can be shorter
    log("=" * 64)
    log(cfg.summary())

    model = FNRModel(cfg)
    c = model.count_params()
    log(f"[model] total={c['total']/1e6:.1f}M trainable={c['trainable']/1e6:.2f}M "
        f"frozen={c['frozen']/1e6:.1f}M ratio={c['ratio']}  "
        f"Fix3_escape={cfg.use_residual_escape}")
    if getattr(cfg, "dual_engine", False):
        log(f"[dual_engine] 50M frozen engine + {c.get('controller', 0)/1e6:.2f}M PathController "
            f"(0-var FunctionCalculator, gain={cfg.dual_fix_gain}); optimizer updates "
            f"controller vars ONLY -> 'train a 50M model with 3.5M'")

    # move model to device BEFORE any forward pass (dtype bench + Trainer)
    dev_probe = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev_probe)

    # auto-dtype: bench fp32 vs fp16 on THIS GPU (Pascal->fp32, Ampere+->fp16)
    if args.dtype == "auto":
        cfg.fp16 = choose_dtype(model, cfg, dev_probe)
    else:
        cfg.fp16 = (args.dtype == "fp16")
    log(f"[dtype] using {'fp16-autocast' if cfg.fp16 else 'fp32'}")

    # data
    train_path = os.path.join(args.data_dir, "train_tokens.pt")
    val_path = os.path.join(args.data_dir, "val_tokens.pt")
    tok_path = os.path.join(args.data_dir, "tokenizer.json")
    for p in (train_path, val_path, tok_path):
        if not os.path.exists(p):
            log(f"[ERROR] missing {p}; run prep_data.py first")
            sys.exit(1)
    tok = BPE.load(tok_path)
    order = None
    if args.curriculum:
        import json as _json
        _cur = _json.load(open(args.curriculum, encoding="utf-8"))
        order = [r["seg"] for r in (_cur.get("easy", []) + _cur.get("novel_tail", []))]
        log(f"[curriculum] easy->hard 段序 {len(order)} 段（来自 {args.curriculum}）")
    train_ds = PackedDataset(train_path, cfg.max_seq_len, order=order)
    val_ds = PackedDataset(val_path, cfg.max_seq_len)
    args.teacher = None   # 本方案不使用 teacher（数据不足靠流水线 + 课程 + 自生成，不蒸馏）
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=0,
                              pin_memory=(torch.cuda.is_available()),
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=0)
    log(f"[data] train={len(train_ds)} val={len(val_ds)} "
        f"(seq={cfg.max_seq_len}, batch={cfg.batch_size})")

    # 细胞分裂感知 resume：若 ckpt 是 mitosis 升级过的（K 已变大），
    # 必须在 Trainer（优化器绑定旧模型参数）之前按新 K 重建模型，
    # 否则 strict 加载会漏掉变形层、冻结体停留在父 K。
    start = 0
    ck = None
    mitosis_rebuild = False
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        mcfg = ck.get("config")
        if mcfg is not None and int(mcfg.get("K", cfg.K)) != int(cfg.K):
            from config import FNRConfig as _Cfg
            from fnr_model import FNRModel as _FNR
            cfg = _Cfg(**mcfg)
            cfg.max_steps = args.max_steps
            cfg.batch_size = args.batch_size
            cfg.max_seq_len = args.seq_len
            model = _FNR(cfg)
            mitosis_rebuild = True
            c = model.count_params()
            log(f"[mitosis-resume] ckpt K={cfg.K} != preset K; 重建模型 "
                f"(total={c['total']/1e6:.1f}M trainable={c['trainable']/1e6:.3f}M)")

    trainer = Trainer(cfg, model, args)

    if ck is not None:
        if mitosis_rebuild:
            model.load_state_dict(ck["model_state_dict"], strict=True)
            from mitosis import restore_split
            restore_split(model, ck.get("mitosis", {}), trainer.device)
            log(f"[mitosis-resume] 严格加载 + M_path 固定基还原 "
                f"(step={ck.get('step')}, mode={ck.get('mitosis', {}).get('mode')})")
        else:
            model.load_state_dict(ck["model_state_dict"], strict=False)
            log(f"[resume] from {args.resume} step={ck.get('step')}")

    dev = trainer.device.type
    log(f"[train] device={dev} fp16={trainer.use_fp16} -> {args.max_steps} steps")
    log("-" * 64)

    gstep = start
    t0 = time.time()
    # 2h 时间预算提示（不启动训练，仅打印）：
    #   实测基线（MX150 单流）：50m B16/S256≈2.7s/step（1.5k tok/s）、
    #   8m B16/S256≈0.94s/step（4.2k tok/s）。双流+多 worker 只省 H2D 间隙
    #   （预计 -3~8%），不改变 GEMM 主体。
    #   2h 可训步数 ≈ 7200s / 单步时长；超出请按此调 --max_steps。
    _toks_2h_50m = int(7200 * 1517 / 1e6)
    _toks_2h_8m = int(7200 * 4164 / 1e6)
    log(f"[2h-budget] 实测基线外推（单卡 MX150）：")
    log(f"  50m (K156, B16/S256, 1.5k tok/s) -> 2h ≈ {_toks_2h_50m}M token "
        f"≈ {int(7200/2.7)} 步（默认 --max_steps {args.max_steps} "
        f"{'在预算内' if args.max_steps <= int(7200/2.7) else '⚠ 超 2h，建议降到 2600'}）")
    log(f"  8m  (K76,  B16/S256, 4.2k tok/s) -> 2h ≈ {_toks_2h_8m}M token "
        f"≈ {int(7200/0.94)} 步（默认 {args.max_steps} "
        f"{'在预算内' if args.max_steps <= int(7200/0.94) else '⚠ 超 2h'}）")
    buf_loss, buf_n = 0.0, 0
    best_val = float("inf")
    tok_seen = 0

    # ── 三阶段自适应：暴力 -> 优化 -> 暴力多数据（--no_phase 关闭）──
    use_phase = (not args.no_phase)
    ACC_MAX = args.grad_accum_max if use_phase else args.grad_accum
    picker = SpeedPhasePicker(ACC_MAX, base_ms=None) if use_phase else None
    cur_acc = ACC_MAX                     # 当前步采用的 micro-batch 数
    carry: list = []                      # 未消化完的 micro-batch（防数据浪费）
    last_phase = None
    if use_phase:
        _ph_desc = f"on (bruteforce->optimize->moredata, acc_max={ACC_MAX})"
    else:
        _ph_desc = f"off (fixed acc={args.grad_accum})"
    log(f"[phase] {_ph_desc}")

    def _epoch_iter(loader, epochs):
        for ep in range(1, epochs + 1):
            if ep > 1:
                log(f"[epoch {ep}/{epochs}] 复用数据池（数据不足时的 L3 手段）")
            for b in loader:
                yield b
    # CPU/GPU 双流流水线（强利用双端）：
    #   CPU：num_workers 并行切窗 + pin_memory（主核不碰数据路径）
    #   GPU：s_copy 后台 H2D 预取下一 batch，s_comp 跑当前步 → 重叠
    use_pipe = (args.num_workers >= 2 and not args.no_pipeline
                and dev == "cuda")
    pipeline = DualStreamPipeline(dev) if use_pipe else None
    _nw = args.num_workers if use_pipe else 0
    _loader = DataLoader(train_ds, batch_size=cfg.batch_size * ACC_MAX,
                         shuffle=(args.curriculum is None),
                         num_workers=_nw, pin_memory=(dev == "cuda"),
                         drop_last=True)
    _it = _epoch_iter(_loader, args.epochs).__iter__()
    if pipeline is not None:
        log(f"[pipeline] CPU/GPU 双流：{_nw} CPU workers + 双缓冲 pinned H2D 重叠 GEMM")
    else:
        log(f"[pipeline] off（num_workers={_nw}，单流直喂）")
    _head = None
    try:
        _head = next(_it)
    except StopIteration:
        _head = None
    if pipeline is not None and _head is not None:
        pipeline.prefetch(_head)
    while _head is not None:
        batch = _head
        try:
            _nxt = next(_it)
        except StopIteration:
            _nxt = None
        if pipeline is not None:
            _g = pipeline.get()
            if _nxt is not None:
                pipeline.prefetch(_nxt)
            batch = _g if _g is not None else batch.to(dev)
        if gstep >= args.max_steps:
            break
        _head = _nxt
        step_t0 = time.time()
        # 每步从"最大档"数据块里取 cur_acc 个 micro-batch 消化；
        # 剩余的放回 carry，下一步继续用 -> 数据 100% 不浪费，且 acc 可动态变。
        micro_all = batch.view(ACC_MAX, cfg.batch_size, cfg.max_seq_len)   # (ACC_MAX, B, S)
        parts = carry + [micro_all[i] for i in range(ACC_MAX)]
        if use_phase:
            cur_acc, ph = picker.acc_for(), picker.current()  # 用上一步 ms 决定的档
        else:
            cur_acc, ph = args.grad_accum, "fixed"
        use_parts = parts[:cur_acc]
        carry = parts[cur_acc:]

        # gradient accumulation window: zero grads once, then cur_acc micro-backwards
        trainer.optimizer.zero_grad(set_to_none=True)
        m = None
        for mi in range(cur_acc):
            m = trainer.train_step(use_parts[mi], gstep,
                                   accumulate=(mi < cur_acc - 1))
        buf_loss += m["loss"]; buf_n += 1
        tok_seen += cfg.batch_size * cfg.max_seq_len * cur_acc
        gstep += 1

        # 把本步实测 ms/step 喂回三阶段调度器 -> 决定下一档 acc（暴力/优化/多数据）
        step_ms = (time.time() - step_t0) * 1000.0
        if use_phase:
            _, ph = picker.record(step_ms, gstep)
            if ph != last_phase:
                log(f"[phase {gstep}] -> {ph} (acc {cur_acc}->{picker.acc_for()}, "
                    f"{step_ms:.0f}ms/step, base {picker.base_ms:.0f}ms)")
                last_phase = ph

        if gstep % args.log_every == 0:
            avg = buf_loss / buf_n
            sec = (time.time() - t0) / gstep
            tps = tok_seen / (sec * gstep)   # 真实累计 tok / 累计秒（反映三阶段动态 acc）
            log(f"[step {gstep}] train={avg:.4f} lr={m['lr']:.2e} "
                f"ent={m['ent']:.3f} {sec*1000:.0f}ms/step "
                f"{tps:.0f}tok/s | total {tok_seen/1e6:.1f}M tok")
            buf_loss, buf_n = 0.0, 0

        if gstep % args.eval_every == 0:
            v = trainer.val_loss(val_loader)
            if v < best_val:
                best_val = v
                log(f"[val {gstep}] val={v:.4f}  (best)")
            else:
                log(f"[val {gstep}] val={v:.4f}  best={best_val:.4f}")

        # save a model file only every save_every steps (default 10000)
        if gstep % args.save_every == 0 and gstep < args.max_steps:
            p = trainer.save_ckpt(gstep)
            log(f"[ckpt {gstep}] saved model file -> {p}")

    # final
    v = trainer.val_loss(val_loader)
    p = trainer.save_ckpt(gstep, val=v, final=True)
    log(f"\n[final] step={gstep} val={v:.4f} -> {p}")
    log("[sample]\n" + trainer.sample_lines(tok, n=6))
    log(f"[done] {gstep} steps in {(time.time()-t0)/3600:.2f}h")


if __name__ == "__main__":
    main()
