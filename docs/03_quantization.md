# 量化流程

## 核心决策：只量化 LLM 主干，ViT 保 FP16

这是本项目最重要的一个技术判断，也是面试必问。**不要背结论，要有数据。**

```bash
python quantize/sensitivity_analysis.py
python benchmark/plot_results.py --sensitivity
```

产出 `results/figures/quant_sensitivity.png`：
X 轴 = 参数量占比（量化收益），Y 轴 = INT8 伪量化 KL 散度（量化代价）。
右下角该量化，左上角不该量化。

**三条理由（按说服力排序）**

1. **ViT 的 outlier 压不住**
   ViT 的 outlier channel 位置随输入图像漂移，per-channel scale 是静态的，
   覆盖不住。LLM 的 outlier channel 相对固定，SmoothQuant 才有效。

2. **收益分配不对等**
   LLM 主干占参数量 ~82%（约 3.1B / 总 3.75B），decode 阶段 memory-bound，权重量化直接转吞吐。
   ViT 只有 ~670M 参数，且只在 prefill 跑一次、是 compute-bound，
   量化它对 TTFT 改善很小，但可能让 OCR / 细粒度识别直接崩。

3. **有实测数据**
   填你自己跑出来的敏感度表。

---

## 支持的量化格式

| 格式 | 说明 | Orin 支持 |
|---|---|---|
| `int8_sq` | SmoothQuant W8A8 | ✅ INT8 Tensor Core |
| `int4_awq` | AWQ W4A16 | ✅ 8GB 板子的唯一选择 |
| `w4a8_awq` | 权重 INT4 + 激活 INT8 | ⚠️ 需验证 SM87 支持 |
| `fp8` | — | ❌ **Orin 是 SM87（Ampere），不支持** |

FP8 需要 Ada（SM89）或 Hopper。别在 Orin 上浪费时间试。

---

## 一、校准集

```bash
python calib/build_vl_calib.py --probe          # 先探测数据源
python calib/build_vl_calib.py --n-samples 512
```

### 1.1 为什么必须是图文混合的

量化校准统计的是每层激活的动态范围。VLM 的 LLM 主干在推理时，输入序列里
很大一部分是**视觉 token**——由 ViT 输出、经 projector 投影而来，
激活分布和文本 embedding 完全不同。

只用纯文本校准，量化 scale 按文本分布定，实际推理时视觉 token 一进来就
大面积截断。**现象是：文字问答正常，一看图就胡说。**

### 1.2 对照实验（本项目的亮点）

```bash
python calib/build_vl_calib.py --text-only-ablation --n-samples 512
```

用纯文本校准集量化一版，和图文混合版对比：

| 校准数据 | 纯文本任务 | 图文任务 | 结论 |
|---|---|---|---|
| 纯文本 | | | |
| 图文混合 | | | |

**如果纯文本版的图文任务掉点明显，你就用数据证明了一个论点，
而不是复述别人的结论。** 这是这个项目最值钱的一条。

### 1.3 数量

128 起步即可出效果，512 是精度/耗时平衡点。做 128/256/512 的消融，
证明你知道边际收益在哪。

### 1.4 数据边界声明

**不得用 MMBench / MMMU / C-Eval 等评测集做校准。**

`calib/build_vl_calib.py` 会记录图片指纹，
`eval/build_eval_set.py` 构建评测集时会用指纹剔除重叠，
最后打印"无重叠，校准/评测分离成立"。

面试官问"你校准和评测是同一批数据吗"，这就是你的答案。

---

## 二、量化执行

```bash
# 在桌面卡上跑（3B 峰值 14-18GB）
python convert/quantize_llm.py \
    --qformat int4_awq \
    --calib calib/data/vl_calib_512.pt \
    --out ckpt/qwen25vl-3b-int4awq
```

### 2.1 排除清单

脚本会自动排除：
- 视觉塔全部（`visual`、`vision_tower`、`merger`、`vision_model`）
- `lm_head`（量化会放大 logits 误差）
- `embed_tokens`（查表操作，量化无收益）
- 各类 norm（逐通道缩放，误差沿层累积）

**换模型时先确认模块名**：
```python
for n, _ in model.named_modules(): print(n)
```
命名对不上，排除规则就失效了，ViT 会被悄悄量化掉。

### 2.2 SmoothQuant 的 α

原理：激活的 outlier 难量化、权重好量化，把难度迁移一部分给权重。

```
Y = (X / s) · (s · W)
s_j = max|X_j|^α / max|W_j|^(1-α)
```

α=0 全留给激活，α=1 全推给权重，通常 0.5~0.8。

**VLM 的最优 α 和纯语言模型不同**，因为视觉 token 的激活分布不一样。
做个扫描：

| α | PPL | MMBench | |
|---|---|---|---|
| 0.5 | | | |
| 0.6 | | | |
| 0.75 | | | |
| 0.85 | | | |

### 2.3 AWQ 的 group_size

128 是通用甜点，64 精度更好但 kernel 更慢。
在 Orin 上建议先试 128，精度不够再降。

---

## 三、精度回归

```bash
python eval/build_eval_set.py --n 100
python eval/eval_consistency.py
```

三个层次：

1. **逐 token 一致性**——贪心解码下，量化前后输出多少 token 后开始分叉。
   最直观："前 N 个 token 完全一致"比一个准确率数字有说服力得多。
2. **任务准确率**——MMBench / TextVQA
3. **分布距离**——首 token logits 的 KL 散度

判读：如果 p10 分叉位置 < 5，说明量化明显改变了模型行为，回去检查。

---

## 四、常见问题

**症状：文字问答正常，一看图就答错**
→ 用纯文本校准了。换图文混合校准集重新量化。

**症状：OCR / 图中文字识别明显变差**
→ ViT 被量化了。检查 `quant_recipe.json` 的 `excluded_modules`。

**症状：量化后完全输出乱码**
→ 不是量化问题，是 M-RoPE。跑 `python runtime/mrope.py` 自检。

**症状：校准时 OOM**
→ 降低 `--max-pixels`。DocVQA 那种高分辨率一张图能吃 4000+ token。
