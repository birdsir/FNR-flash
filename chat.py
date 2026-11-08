"""
FNR-100M 中文对话界面
用法:
  python chat.py --checkpoint ./checkpoints/final.pt
  python chat.py --checkpoint ./checkpoints/final.pt --interactive
  python chat.py --checkpoint ./checkpoints/final.pt --prompt "你好" --max_new 64

格式: 用 "用户：...\n助手：..." 做对话模板, 模型续写助手回复。
"""
import sys, os, re, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from config import FNRConfig
from fnr_model import FNRModel
from tokenizer import BPE, BOS


SYSTEM = ("你是一名乐于助人的中文助手。\n")


def load_model(ckpt_path: str, device: str = "cuda") -> tuple:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = FNRConfig(**ckpt["config"])
    model = FNRModel(cfg)
    missing = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval().to(device)
    return model, cfg, ckpt.get("tokenizer_vocab", None)


def build_prompt(tok, system: str, user_msgs, assistant_msg):
    text = system + assistant_msg + user_msgs
    ids = tok.encode(text, add_bos=True)
    return torch.tensor([ids], dtype=torch.long)


def stop_decode(tok, ids, stop_words):
    """Truncate at first stop word / newline sequence after generated part."""
    s = tok.decode(ids)
    for sw in stop_words:
        idx = s.find(sw, 1)
        if idx > 0:
            s = s[:idx]
    return s.strip()


@torch.no_grad()
def chat_once(model, cfg, tok, history, user_msg, device,
              max_new=96, temperature=0.85, top_k=60, do_sample=True,
              stop_words=None, use_graph=True):
    if stop_words is None:
        stop_words = ["\n用户：", "用户：", "\n\n"]
    # format: user lines
    user_lines = "".join(f"用户：{h}\n助手：{a}\n" for h, a in history)
    user_lines += f"用户：{user_msg}\n助手："
    x = build_prompt(tok, SYSTEM, user_lines, "")
    x = x.to(device)

    # Theoretical-limit decode: CUDA-Graph path (B=1, ~4x faster), fall back
    # to the KV-cache eager path on any capture failure.
    if use_graph and x.size(0) == 1 and device == "cuda":
        try:
            out_ids = model.generate_graph(x, max_new_tokens=max_new,
                                           temperature=temperature, top_k=top_k,
                                           do_sample=do_sample,
                                           max_len=max(len(x.size(1)) + max_new,
                                                       cfg.max_seq_len))
        except Exception:
            out_ids = model.generate(x, max_new_tokens=max_new,
                                     temperature=temperature, top_k=top_k,
                                     do_sample=do_sample)
    else:
        out_ids = model.generate(x, max_new_tokens=max_new,
                                 temperature=temperature, top_k=top_k,
                                 do_sample=do_sample)
    gen = out_ids[0, x.shape[1]:]
    text = stop_decode(tok, gen.tolist(), stop_words)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="./checkpoints/final.pt")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--max_new", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top_k", type=int, default=60)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--no_graph", action="store_true",
                    help="disable CUDA-Graph decode (fall back to KV-cache eager)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, _ = load_model(args.checkpoint, device)
    tok = BPE.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "corpus", "tokenizer.json"))
    counts = model.count_params()
    print(f"[model] total={counts['total']/1e6:.1f}M trainable={counts['trainable']/1e6:.2f}M")

    history = []
    if args.prompt:
        user = args.prompt
        reply = chat_once(model, cfg, tok, history, user, device,
                          max_new=args.max_new, temperature=args.temperature,
                          top_k=args.top_k, do_sample=not args.greedy,
                          use_graph=not args.no_graph)
        print(f"用户：{user}\n助手：{reply}")
        return

    if not args.interactive:
        args.interactive = True
    print("FNR-100M 对话模式。输入 'exit' 退出, 'clear' 清屏。\n")
    while True:
        try:
            user = input("用户：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user in ("exit", "quit", "q"):
            break
        if user == "clear":
            history.clear()
            print("(已清空对话历史)")
            continue
        reply = chat_once(model, cfg, tok, history, user, device,
                          max_new=args.max_new, temperature=args.temperature,
                          top_k=args.top_k, do_sample=not args.greedy,
                          use_graph=not args.no_graph)
        print(f"助手：{reply}\n")
        history.append((user, reply))
        if len(history) > 6:
            history.pop(0)
    print("再见。")


if __name__ == "__main__":
    main()
