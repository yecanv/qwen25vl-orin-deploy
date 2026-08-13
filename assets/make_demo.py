#!/usr/bin/env python3
"""
生成 benchmark 用的测试图。

为什么不直接放一张图：benchmark 需要可控的分辨率和内容复杂度，
而且随包分发真实照片有版权问题。这里程序化生成，可复现。

用法：python assets/make_demo.py           # 生成默认 3 张
     python assets/make_demo.py --size 1280 720
"""
import argparse
from pathlib import Path


def make_scene(w, h, seed=0):
    """生成一张有一定视觉复杂度的合成图（模拟车载前视场景的结构）。"""
    from PIL import Image, ImageDraw
    import random
    rng = random.Random(seed)
    img = Image.new("RGB", (w, h), (135, 180, 220))      # 天空
    d = ImageDraw.Draw(img)

    d.rectangle([0, int(h * 0.6), w, h], fill=(90, 90, 95))          # 路面
    d.polygon([(w*0.35, h*0.6), (w*0.65, h*0.6), (w, h), (0, h)],
              fill=(105, 105, 110))                                  # 透视车道
    for i in range(6):                                               # 车道线
        y = h * (0.62 + i * 0.06)
        cw = w * (0.02 + i * 0.012)
        d.rectangle([w/2 - cw/2, y, w/2 + cw/2, y + h*0.015], fill=(230, 230, 210))

    for i in range(rng.randint(3, 6)):                               # 建筑
        bx = rng.randint(0, int(w*0.9)); bw = rng.randint(int(w*0.06), int(w*0.15))
        bh = rng.randint(int(h*0.15), int(h*0.4))
        d.rectangle([bx, h*0.6 - bh, bx+bw, h*0.6],
                    fill=(rng.randint(90,150),)*3)
        for r in range(int(bh//(h*0.05))):                           # 窗户，增加高频细节
            for c in range(int(bw//(w*0.02))):
                if rng.random() > 0.4:
                    wx = bx + c*w*0.02 + w*0.005
                    wy = h*0.6 - bh + r*h*0.05 + h*0.01
                    d.rectangle([wx, wy, wx+w*0.01, wy+h*0.025],
                                fill=(250, 230, 150))

    for i in range(rng.randint(2, 4)):                               # 车辆
        cx = rng.randint(int(w*0.1), int(w*0.8)); cy = rng.randint(int(h*0.62), int(h*0.85))
        cw = rng.randint(int(w*0.06), int(w*0.12)); ch = int(cw*0.55)
        d.rounded_rectangle([cx, cy, cx+cw, cy+ch], radius=int(cw*0.08),
                            fill=(rng.randint(60,200), rng.randint(60,200), rng.randint(60,200)))
        d.rectangle([cx+cw*0.15, cy+ch*0.1, cx+cw*0.85, cy+ch*0.5], fill=(40,50,70))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", nargs=2, type=int, default=None)
    ap.add_argument("--out-dir", default="assets")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if args.size:
        sizes = [("custom", args.size[0], args.size[1])]
    else:
        # 三档分辨率，对应 bench_latency.py 的扫描档位
        sizes = [("demo_low", 448, 448), ("demo", 896, 896), ("demo_high", 1792, 1008)]

    for name, w, h in sizes:
        p = out / f"{name}.jpg"
        make_scene(w, h, seed=hash(name) & 0xFFFF).save(p, quality=92)
        n_tok = (w//14) * (h//14) // 4
        print(f"{p}  {w}x{h}  预计视觉 token ≈ {n_tok}")


if __name__ == "__main__":
    main()
