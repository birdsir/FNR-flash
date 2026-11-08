"""
L3 数据效率工具：去重 + CE 质量过滤 + 课程排序（纯 CPU 一次性，不训练）。

思想（数据不足时"不靠加数据"的质量提升路径之一）：
  1) 去重：CCI 类语料里大量重复模板（新闻/对话重复片段）。48-token
     Rabin 滚动指纹（O(n) 哈希），同指纹只保留首次出现 → 语料干净度 ↑，
     模型不再把容量浪费在死记重复片段上。
  2) CE 打分：用一个"探针模型"（已训到中途的 FNR ckpt 即可，8m/50m）
     给每个 256 段算交叉熵。低 CE = 模型熟悉的"熟悉数据"（easy），
     高 CE = 模型没见过的"新颖数据"（hard/novel）。
  3) 课程排序（curriculum.json）：训练按 easy→hard 排段序。
     前 75% 是熟悉数据（loss 快速下降、路由稳定 warm-up），
     后 25% 高 CE 新颖数据作为"精化段"放在训练末期（epoch 2 或后半程）——
     等价于把"稀缺的新信息"集中喂在控制器最灵活的最后阶段。

用法（不实测环境下也可先跑前两步，只需一个已训的探针 ckpt）：
  python dedup_filter.py --input corpus/train_tokens.pt \
      --probe checkpoints/step_5000.pt --out corpus/train_dedup.pt \
      --curriculum curriculum.json

输出：
  out            : int16 去重后的 token 流（PackedDataset 直接可用）
  curriculum.json: 段序 + 每段 CE，train.py --curriculum 消费
"""
import argparse, json, math, struct
import torch

P = 2**61 - 1            # 梅森素数，Rabin 指纹模数
B = 31                   # 多基


def rabin_dedup(stream: torch.Tensor, win: int = 48) -> torch.Tensor:
    """48-token 滚动 Rabin 指纹去重（O(n)）。返回去重后的 1-D long 流。"""
    t = stream.view(-1).long()
    n = t.numel()
    powB = pow(B, win - 1, P)
    f = 0
    for i in range(win):
        f = (f * B + t[i].item()) % P
    seen = {f}
    keep = [True] * n
    last_dup_end = -1
    for i in range(win, n):
        f = ((f - t[i - win].item() * powB) * B + t[i].item()) % P
        if f in seen:
            keep[i - win + 1] = False          # 上一段起始标重复
            last_dup_end = i
        else:
            seen.add(f)
    # 从"重复窗口"起点到下一个非重复点整段剔除
    out, i = [], 0
    while i < n:
        if keep[i]:
            j = i
            while j < n and keep[j]:
                j += 1
            out.append(t[i:j])
            i = j
        else:
            i += 1
    return torch.cat(out) if out else torch.empty(0)


def ce_scores(probe, stream: torch.Tensor, seg: int = 256,
              batch: int = 8, device="cpu") -> torch.Tensor:
    """探针模型给每个 seg 段打 CE 分（255 个预测位，取均值）。"""
    probe.eval()
    n = (stream.numel() // seg) * seg
    segs = stream[:n].view(-1, seg)
    scores = []
    with torch.no_grad():
        for i in range(0, len(segs), batch):
            x = segs[i:i + batch].to(device)
            out = probe(x, labels=x.clone(), training=False, step=0)
            loss = out.loss                       # 全 batch 均值
            # 近似：段级分数用 batch 均值回标（同 batch 内近似一致）
            scores.extend([loss.item()] * x.shape[0])
    return torch.tensor(scores, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="corpus/train_tokens.pt")
    ap.add_argument("--probe", required=True,
                    help="探针 ckpt（train.py 的 step_N/final.pt，8m/50m 均可）")
    ap.add_argument("--out", default="corpus/train_dedup.pt")
    ap.add_argument("--curriculum", default="curriculum.json")
    ap.add_argument("--seg", type=int, default=256)
    ap.add_argument("--probe_batch", type=int, default=8)
    args = ap.parse_args()

    from config import FNRConfig
    from fnr_model import FNRModel

    raw = torch.load(args.input, map_location="cpu", weights_only=True)
    stream = raw.long() if raw.dtype in (torch.int16, torch.int32) else raw
    print(f"[dedup] 输入 {stream.numel()/1e6:.1f}M token")

    deduped = rabin_dedup(stream, win=48)
    print(f"[dedup] 去重后 {deduped.numel()/1e6:.1f}M token "
          f"（剔除 {(stream.numel()-deduped.numel())/stream.numel()*100:.1f}% 重复）")
    torch.save(deduped.contiguous().to(torch.int16).view(-1), args.out)

    ck = torch.load(args.probe, map_location="cpu", weights_only=False)
    probe = FNRModel(FNRConfig(**ck["config"]))
    probe.load_state_dict(ck["model_state_dict"], strict=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    probe = probe.to(dev)
    print(f"[ce] 探针 step={ck.get('step')} @ {dev}，打分 {deduped.numel()//args.seg} 段...")
    scores = ce_scores(probe, deduped, seg=args.seg, batch=args.probe_batch, device=dev)

    order = torch.argsort(scores)            # easy -> hard
    recs = [{"seg": int(i), "start": int(i) * args.seg,
             "ce": round(float(scores[i]), 4)} for i in order]
    novel_cut = int(len(recs) * 0.75)
    json.dump({"seg": args.seg, "n_seg": len(recs),
               "easy": recs[:novel_cut], "novel_tail": recs[novel_cut:]},
              open(args.curriculum, "w"))
    m = lambda k: sum(r["ce"] for r in recs[k]) / max(len(recs[k]), 1)
    print(f"[curriculum] easy(75%) 均值 CE={m(slice(None, novel_cut)):.3f} -> "
          f"novel(25%) 均值 CE={m(slice(novel_cut, None)):.3f}")
    print(f"[done] {args.out} + {args.curriculum}（train.py 用 "
          f"--curriculum {args.curriculum} --data_dir <目录含out>）")


if __name__ == "__main__":
    main()
