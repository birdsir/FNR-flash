"""
采样评估：从 checkpoint 生成中文样本，衡量模型"像不像话"。
用法:
  python sample.py --checkpoint ./checkpoints/final.pt --n 8
输出若干续写样本 + 对话样本, 供人工判断模型是否"健康"。
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from config import FNRConfig
from fnr_model import FNRModel
from tokenizer import BPE, BOS


SEEDS = [
    "今天我去公园散步，",
    "中国的首都是",
    "我喜欢在夏天吃",
    "人工智能技术正在",
    "用户：你好\n助手：",
    "用户：帮我写一句祝福\n助手：",
    "用户：失眠怎么办\n助手：",
    "《红楼梦》的作者是",
    "春天到了，",
    "学习一门语言最有效的方法是",
]


def load(ckpt: str, device: str):
    c = torch.load(ckpt, map_location=device)
    cfg = FNRConfig(**c["config"])
    m = FNRModel(cfg)
    m.load_state_dict(c["model_state_dict"], strict=False)
    m.eval().to(device)
    return m, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=len(SEEDS))
    ap.add_argument("--max_new", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--no_graph", action="store_true",
                    help="disable CUDA-Graph decode (use KV-cache eager instead)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load(args.checkpoint, device)
    tok = BPE.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "corpus", "tokenizer.json"))
    counts = model.count_params()
    print(f"[model] total={counts['total']/1e6:.1f}M trainable={counts['trainable']/1e6:.2f}M")
    print("=" * 60)

    for i, seed in enumerate(SEEDS[:args.n]):
        ids = tok.encode(seed, add_bos=True)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        if not args.no_graph and device == "cuda":
            try:
                out = model.generate_graph(x, max_new_tokens=args.max_new,
                                           temperature=args.temperature,
                                           top_k=args.top_k, do_sample=True,
                                           max_len=max(len(ids) + args.max_new,
                                                       cfg.max_seq_len))
            except Exception:
                out = model.generate(x, max_new_tokens=args.max_new,
                                     temperature=args.temperature,
                                     top_k=args.top_k, do_sample=True)
        else:
            out = model.generate(x, max_new_tokens=args.max_new,
                                 temperature=args.temperature, top_k=args.top_k,
                                 do_sample=True)
        gen = tok.decode(out[0, x.shape[1]:])
        # clean: cut at a clean break
        print(f"\n[{i}] {seed}\n→ {gen[:160].strip()}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
