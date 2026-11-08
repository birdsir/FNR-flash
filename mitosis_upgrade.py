"""
细胞分裂式升级 CLI（"计算改权重"，0 训练，秒级完成）。

用法（升级 = 一次权重拷贝，不启动任何训练）：
  # 从 50m 双引擎 ckpt 分裂出 r1（K 156 -> 312，互补模式 = 子槽位真特化）
  python mitosis_upgrade.py --in checkpoints/50m/final.pt \
      --out checkpoints/50m/final_r1.pt --mode complement

  # 链式升级 r1 -> r2（K 312 -> 624）：
  python mitosis_upgrade.py --in checkpoints/50m/final_r1.pt \
      --out checkpoints/50m/final_r2.pt --mode complement

完成后打印二阶段短重训命令（只训 <=3.5M 控制器，50M 冻结体 0 梯度）：
  python train.py --preset 50m --resume checkpoints/50m/final_r1.pt \
      --max_steps 1500 --batch_size 8 --seq_len 256

模式：
  complement（默认）：子算子 = 父算子的互补通道分割（P_a+P_b=I，0 变量），
      产生"半通道"新算子 → 冻结库获得新的组合能力（跨父通道混合），
      代价：warm-start 近似（算子贡献 ~0.5 衰减，二阶段短重训消化）。
  duplicate（零漂移）：子 = 父精确复制，路由 top-k 归一化下 Σw·f = f 构造性
      恒等（偏差 <1e-5），行为完全不变，只增加路由分辨率；
      适合"先分裂保行为、之后再加 LoRA/变量逐步分化"的保守路线。
"""
import argparse, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FNRConfig
from fnr_model import FNRModel
from mitosis import cell_divide_all, verify_parent_reachable, serialize_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True,
                    help="源 ckpt（train.py 的 step_N/final.pt）")
    ap.add_argument("--out", required=True, help="升级后 ckpt 路径")
    ap.add_argument("--mode", default="complement",
                    choices=["complement", "duplicate"],
                    help="complement=子槽位特化(近似warm-start) "
                         "duplicate=零漂移(行为精确不变)")
    ap.add_argument("--seed", type=int, default=2026,
                    help="分裂正交基种子（0 可训练变量，固定程序）")
    args = ap.parse_args()

    ck = torch.load(args.inp, map_location="cpu", weights_only=False)
    cfg = FNRConfig(**ck["config"])
    K_old = cfg.K

    t0 = time.time()
    model = FNRModel(cfg)
    model.load_state_dict(ck["model_state_dict"], strict=True)  # 分裂前严格对齐
    new_K = cell_divide_all(model, mode=args.mode, seed=args.seed)
    # 静态自检（不训练）：父算子输出在 0.5/0.5 子槽位选取时的可达偏差
    err = verify_parent_reachable(model.blocks[0].operator_bank,
                                  mode=args.mode, seed=args.seed)
    c = model.count_params()
    dt = time.time() - t0
    print(f"[mitosis] K {K_old} -> {new_K} (mode={args.mode}, seed={args.seed}) "
          f"纯权重拷贝耗时 {dt:.2f}s（0 训练）")
    print(f"[mitosis] 父算子可达性偏差 = {err:.3e}"
          + ("（duplicate 构造性恒等 <1e-5 OK）" if args.mode == "duplicate"
             else "（complement: GELU 下 ~0.5 衰减，二阶段短重训消化）"))
    print(f"[mitosis] 参数账: total={c['total']/1e6:.1f}M "
          f"trainable={c['trainable']/1e6:.3f}M frozen={c['frozen']/1e6:.1f}M "
          f"ratio={c['ratio']}")
    if c["trainable"] > 3.5e6:
        print(f"[mitosis] ⚠ 可训练核 {c['trainable']/1e6:.2f}M 超过 3.5M 预算，"
              "建议二阶段只用控制器 + 更低 dual_hidden")

    # 保存升级 ckpt（config 已是分裂后 K；mitosis 段由 serialize_split 打包，
    # 携带 0 变量 M_path 固定基，使 train.py --resume 能严格重建分裂瞬间直通项）
    d = {"step": ck.get("step", 0),
         "val_loss": ck.get("val_loss"),
         "model_state_dict": model.state_dict(),
         "config": cfg.__dict__,
         "mitosis": serialize_split(model, mode=args.mode, seed=args.seed,
                                    K_parent=K_old)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(d, args.out)
    print(f"[mitosis] 升级 ckpt -> {args.out}")
    print("\n二阶段短重训（只训 <=3.5M 控制器；50M 冻结体 0 梯度）：")
    print(f"  python train.py --preset 50m --resume {args.out} "
          f"--max_steps 1500 --batch_size 8 --seq_len 256")
    print("链式升级（r2）：把 --in 指向本输出再跑一次即可（K 再翻倍）。")


if __name__ == "__main__":
    main()
