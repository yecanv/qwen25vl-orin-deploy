# 自补 M-RoPE 到 TRT-LLM 0.12 的可行性研判(源码级)

> 结论:技术可行但不划算——外置 RoPE(周级、KV cache 簿记复杂、不支持分支)vs
> 刷 JetPack 7 走 Edge-LLM 官方支持(半天、已知可行)。研判本身把"0.12 缺 mrope"
> 从文档引用升级为 plugin 签名级判断。

## 源码证据(对比 v0.12.0-jetson vs v0.15.0 的 functional.py)

| 事实 | 证据 |
|---|---|
| 0.12 的 gpt_attention **有** rotary_cos_sin / rotary_inv_freq 输入 | func L4504-4505、L5024-5027 |
| 但 rotary_cos_sin 是**常量缓存,按一维位置在 kernel 内索引** | 0.12 docstring:"reused among different requests. It is taken as constant tensor" |
| → 无法把每 token 三维位置 (t,h,w) 烤进这张表 | 常量缓存 cache[i] 与 token 的 (t,h,w) 无关 |
| 0.15 的 functional.py **也零个 mrope** | grep mrope = 0 |
| → mrope 不在底层 attention plugin,而在更上层模型定义/新执行流 | 两版底层签名一致,差异在上层 |

## 三条子路难度谱

- **A1-便宜版(换 rotary_cos_sin 表)**:❌ 已证伪。常量缓存按一维位置索引,三维烤不进。
- **A1-可行版(模型定义层外置 RoPE)**:⚠ 关内核 rope(rotary_dim=0),在 Python 模型定义里
  自己给 Q/K 施加三维 mrope 旋转再喂 plugin。两个硬骨头:①KV cache 存旋转后的 K;
  ②decode 每步新 token 按 mrope_position_delta 续算并现场旋转——正是 0.15 上游补的那套。
  无 CUDA 手术,但周级、每改重建引擎、在不支持分支上做。
- **A2(CUDA 内核移植)**:❌❌ 4h 重编译 + 版本矩阵 + 在不支持分支重做官方已做的事。

## 与刷 JP7 的对比

| 维度 | 自补 A1-可行版 | 刷 JP7 + Edge-LLM |
|---|---|---|
| 工期 | 周级 | 半天 |
| 风险 | 高(KV簿记/不支持分支) | 低(官方支持 Qwen2.5-VL 含 AWQ) |
| 交付确定性 | 不确定 | 官方正路 |
| 学习价值 | 极高(读内核、懂 mrope 落点) | 中(用新工具) |

## 结论要点

「我读过 0.12 和 0.15 的 gpt_attention 签名:0.12 有 rotary_cos_sin 输入但它是按一维位置
索引的常量缓存,烤不进三维;补 mrope 得在模型定义层外置 RoPE 并自己处理 decode 位置续算,
也就是 0.15 上游补的那套。所以升级 JP7 走 Edge-LLM 比反向移植划算——这是我读源码得出的判断,
不是文档转述。」
