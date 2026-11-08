"""
流式数据流水线：CCI parquet（data/*.parquet, 列 id+content）+ 合成对话语料
-> 训练 BPE(8192) -> 边读边打包成 int16 token 流。

内存策略（机器 8GB）：
  1) 随机采样 CCI 的 content 前若干行 -> 训练 BPE 词表
  2) 按 parquet 行组(1万行)流式迭代 content 列, 逐行 encode + 累积到 token 上限
  3) 混入少量合成对话语料（教会 用户:/助手: 格式）
  4) 落盘 int16 token 流（词表 8192 < 32767, 内存减半）

用法:
  python prep_data.py                          # 默认 token 预算 60M
  python prep_data.py --token_budget 100000000 --cci_rows 1000000
  python prep_data.py --list                   # 只统计可用行数
"""
import os, re, json, random, argparse, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import BPE

BASE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(BASE, "corpus")
DATA = os.path.join(BASE, "data")
random.seed(42)

# ── 1. 定位 CCI parquet ──
def find_cci():
    files = []
    for d in (DATA, CORPUS, BASE):
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.lower().endswith(".parquet"):
                files.append(os.path.join(d, fn))
    return sorted(files)


def iter_cci_content(rows=0, batch=10000, content_col="content"):
    """Yield (content, id) one row at a time from all CCI parquets, streaming."""
    import pyarrow.parquet as pq
    f = find_cci()
    if not f:
        return
    count = 0
    for path in f:
        pf = pq.ParquetFile(path)
        cols = [content_col]
        total_rows = pf.metadata.num_rows
        n_rows = total_rows if rows == 0 else min(rows, total_rows)
        print(f"[cci] {os.path.basename(path)}: {total_rows:,} rows, "
              f"reading {n_rows:,}", flush=True)
        for batch_ in pf.iter_batches(batch_size=batch, columns=["content", "id"]):
            cont = batch_["content"].to_pylist()
            ids = batch_["id"].to_pylist()
            for c, i in zip(cont, ids):
                c = (c or "").strip()
                if c:
                    yield c, i
                    count += 1
                if rows and count >= rows:
                    return
    print(f"[cci] total usable rows: {count:,}", flush=True)


# ── 2. 采样 content 训 BPE（内存安全：只留一个字符池）──
def sample_for_bpe(max_chars=6_000_000):
    pool = []
    total = 0
    for c, _ in iter_cci_content():
        pool.append(c)
        total += len(c)
        if total >= max_chars:
            break
    print(f"[bpe] sampled {len(pool):,} rows / {total:,} chars", flush=True)
    return pool


# ── 3. 合成对话语料（教会 用户:/助手: 格式）──
TOPICS = [
    ("天气", ["今天天气怎么样", "明天会下雨吗", "北京现在几度", "这个季节适合穿什么"],
     ["建议您出门前查看当地预报。今天多云转晴，气温适宜，早晚偏凉，适合加件外套。",
      "明天可能有阵雨，出门记得带伞。", "早晚温差大，建议薄外套加长袖。"]),
    ("美食", ["有什么好吃的推荐", "川菜有什么特点", "怎么包饺子", "火锅用什么汤底好"],
     ["川菜以麻辣见长，常用花椒、干辣椒、豆瓣酱。经典菜有麻婆豆腐、水煮鱼、回锅肉。",
      "把面粉加水揉成团，擀成圆皮放肉馅或对折捏紧即可。", "锅底推荐番茄或菌汤，清淡又鲜美。"]),
    ("学习", ["怎么提高记忆力", "学英语有什么好方法", "怎么制定学习计划", "孩子不爱读书怎么办"],
     ["可以试试间隔复习法：学一遍，一天后复习，再三天后复习，重复三次基本就能记住。",
      "每天听一首英文歌并背两句歌词，循序渐进更有效。", "把大目标拆成每天可执行的小步骤。",
      "先从孩子感兴趣的领域入手，比如科学或动物，慢慢培养阅读习惯。"]),
    ("健康", ["失眠怎么办", "每天睡多久合适", "久坐有什么危害", "怎么缓解眼睛疲劳"],
     ["睡前避免蓝光、保持卧室安静凉爽，有助入眠。若长期失眠建议就医。",
      "成人每天建议睡七到九小时，规律作息比偶尔补觉更重要。",
      "久坐会影响血液循环，建议每隔一小时起身活动五分钟。", "每用眼二十分钟看远处六秒，做做眼保健操。"]),
    ("旅行", ["去云南玩几天合适", "旅游要带什么", "国庆去哪人少", "第一次去西藏注意什么"],
     ["云南建议五到七天，可安排昆明、大理、丽江一线。",
      "带身份证、常用药、防晒、舒适鞋子即可。",
      "避开热门城市，去一些小众古镇，体验更好。",
      "注意高原反应，别剧烈运动，多喝水慢慢适应。"]),
    ("科技", ["什么是人工智能", "手机内存不够怎么办", "什么是5G", "电脑发热正常吗"],
     ["人工智能是让机器模拟人类智能的技术，包括语音识别、图像理解、自动决策等，应用很广。",
      "可以清理缓存、删掉不常用应用，或用云相册同步到云端。",
      "5G是第五代移动通信，速度更快延迟更低，适合大数据场景。",
      "轻度发热正常，长期过高要检查散热，必要时清灰。"]),
    ("历史", ["唐朝什么时候建立", "秦始皇统一了哪些地方", "丝绸之路通向哪里", "四大发明是什么"],
     ["唐朝建立于公元618年，唐太宗李世民开创了贞观之治，是中国历史上的鼎盛时期之一。",
      "秦始皇统一六国，并修长城、统一度量衡，奠定了中国统一的基础。",
      "丝绸之路从长安出发，通往中亚、西亚，连接东西方贸易与文化交流。",
      "四大发明是造纸术、印刷术、指南针、火药，深刻改变了世界文明进程。"]),
    ("情感", ["朋友不理我怎么办", "工作压力大怎么调节", "和伴侣吵架了怎么办", "最近心情不好"],
     ["先给彼此空间，冷静后再找机会好好沟通，表达感受而不是指责。",
      "可以做深呼吸、运动或找朋友聊聊；若持续影响生活，考虑寻求专业帮助。",
      "吵架后先冷静，就事论事沟通，避免翻旧账，多换位思考。",
      "心情不好很正常，做点让自己放松的事，必要时找人倾诉。"]),
    ("写作", ["帮我写一句生日祝福", "写一段鼓励我的话", "帮我想个创意", "写一句座右铭"],
     ["愿你的每一天都被温柔以待，生日快乐，未来可期。",
      "即使前路漫漫，每一步都算数，你比自己想象的更强大，继续加油。",
      "把不同领域的概念组合起来，比如用音乐设计一个故事节奏。",
      "「日拱一卒，功不唐捐」，适合放在桌前激励自己。"]),
    ("生活", ["怎么省电", "衣服怎么收纳省空间", "怎么养成早睡习惯", "养什么植物好打理"],
     ["空调设26度、用LED灯、拔掉待机电器插头，能明显省电。",
      "用真空袋压羽绒服、统一收纳盒、利用垂直空间，最省空间。",
      "固定就寝时间，睡前一小时放下手机，保持卧室黑暗安静，几周就能养成。",
      "绿萝、虎皮兰都很好打理，浇水少、耐阴。"]),
]
FILLERS = ["好的，", "关于", "简单说，", "我的看法是，", "其实", "总的来说，"]
STM = [
    "生活由许多平凡而美好的瞬间组成。", "坚持每天进步一点点，累积起来就是巨大的改变。",
    "真正的勇敢，是看清生活的样子之后依然热爱它。", "好的关系建立在相互理解与尊重之上。",
    "把复杂的问题拆成小步，每一步都能带来成就感。", "阅读是最低成本、最高回报的投资。",
    "保持好奇心，世界会给你更多答案。", "健康是1，其他都是后面的0。",
]


def gen_dialog_lines(n=8000):
    lines = []
    for _ in range(n):
        cat, qs, ans = random.choice(TOPICS)
        q = random.choice(qs); a = random.choice(ans)
        lines.append(f"用户：{q}\n助手：{random.choice(FILLERS)}{a}\n")
        if random.random() < 0.4:
            lines.append(f"问：{q}\n答：{a}\n")
    for _ in range(n // 4):
        lines.append(random.choice(STM) + "\n")
    random.shuffle(lines)
    return lines


# ── 4. 流式打包到 int16 token 流 ──
def pack_stream(tok, token_budget, cci_rows, mix_dialog=6000):
    import torch
    acc = []
    lines = 0
    for c, _ in iter_cci_content(rows=cci_rows):
        ids = tok.encode(c)
        if len(ids) > 2:
            acc.extend(ids)
            acc.append(2)
        lines += 1
        if len(acc) >= token_budget:
            break
    # 混入对话语料（教会对话格式；放在末尾不影响主分布）
    dlines = gen_dialog_lines(mix_dialog)
    for t in dlines:
        ids = tok.encode(t)
        if len(ids) > 2:
            acc.extend(ids)
            acc.append(2)
    print(f"[pack] CCI lines={lines:,} dialog={len(dlines):,} "
          f"tokens={len(acc):,} (budget {token_budget:,})", flush=True)

    n_val = int(len(acc) * 0.005)
    val, train = acc[-n_val:], acc[:-n_val]
    torch.save(torch.tensor(train, dtype=torch.int16),
               os.path.join(CORPUS, "train_tokens.pt"))
    torch.save(torch.tensor(val, dtype=torch.int16),
               os.path.join(CORPUS, "val_tokens.pt"))
    print(f"[pack] train={len(train):,} val={len(val):,} tokens saved (int16)",
          flush=True)
    rt = tok.decode(tok.encode("你好，今天天气很好"))
    print(f"[selfcheck] roundtrip {'OK' if rt == '你好，今天天气很好' else 'MISMATCH'} -> {rt!r}",
          flush=True)
    return len(train)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token_budget", type=int, default=200_000_000,
                   help="8M sweet-spot default ~160-200M tokens (Chinchilla 20*P); "
                        "raise to 1B+ only with more CCI shards")
    ap.add_argument("--cci_rows", type=int, default=0,
                    help="max CCI rows to read (0 = all)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    cci = find_cci()
    print(f"[find] CCI parquet: {[os.path.basename(x) for x in cci]}")
    if not cci:
        print("[ERROR] no data/*.parquet found")
        sys.exit(1)
    if a.list:
        n = sum(1 for _ in iter_cci_content(rows=0))
        print(f"[list] total usable rows: {n:,}")
        return

    # 1) 采样训 BPE
    pool = sample_for_bpe(max_chars=6_000_000)
    pool = pool[:80_000]
    sample = "\n".join(random.sample(pool, min(200_000, len(pool))))
    tok = BPE.train(sample, vocab_size=8192, char_budget=4000, verbose=True)
    tok.save(os.path.join(CORPUS, "tokenizer.json"))

    # 2) 流式打包
    pack_stream(tok, a.token_budget, a.cci_rows)


if __name__ == "__main__":
    main()
