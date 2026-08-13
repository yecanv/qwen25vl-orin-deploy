# 混合端到端:TRT ViT → llama.cpp LLM(公开 API 集成)✅ 已实测通过

> **2026-08-12 板上实测 PASS**:手写 C++ 驱动(runtime/hybrid_driver.cpp,~170行,零改
> llama.cpp 源码)一次编译一次运行,双图判据全过(与 HF 参考逐物一致、零串扰)。
> **TTFT ≈1.9s(741ms ViT + 1133ms prefill,分段实测之和)vs 纯 llama.cpp 链 3.59s,
> 提速约 1.7~1.9 倍**(口径:未含预处理,对照值含;详见 results/.../hybrid_llamacpp_e2e.json)。
> decode 25.8~26.0 tok/s 与纯链一致(LLM 同为 Q4_K_M,无回退)。
> 一箭三雕全部兑现:端到端(绕开 TRT-LLM mrope)+ 快且正确的视觉(照片乱码 bug 从链路
> 消失)+ 板上 JP6.2 原生(零刷机零 fork)。

## 注入口证据(llama.cpp master, include/llama.h)

```c
typedef struct llama_batch {         // L256
    llama_token  *  token;           // 喂 token id
    float        *  embd;            // L260 ← 喂 token embeddings(token 为 NULL 时使用)
    llama_pos    *  pos;             // L261 ← 自定义位置(mrope 用)
    ...
} llama_batch;
// L245 注释:embd : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
```

- **mtmd 高层接口**:不认外部 embedding(mtmd.h 只有 mtmd_get_output_embd 取出,无注入)。
- **llama.cpp 核心**:llama_batch.embd 是公开注入口——绕开 mtmd,直接驱动 llama_batch
  即可把 TRT ViT 特征喂进 LLM,**不改 llama.cpp 源码**,是集成不是 fork。

## 架构

```text
TRT ViT 引擎(TensorRT 运行时)  750ms,golden 0.999853
   ↓ 视觉特征 [1024, 2048](已含 merger 投影到 LLM 隐层维)
自定义驱动:构造 llama_batch
   - 视觉位:embd=TRT特征, token=NULL, pos=mrope 三维位置(4值/token 布局)
   - 文本位:token=文本 id
   ↓ llama_decode
llama.cpp LLM(Q4_K_M, 26.55 tok/s):自带 mrope,端到端本就通
```

## 一箭三雕

1. **端到端**:llama.cpp LLM 自带 mrope,完全不需要 TRT-LLM 的 mrope 支持。
2. **快且正确的视觉**:TRT ViT 3.46 倍且 golden 验证;视觉换成 TRT 后,
   llama.cpp 的 GPU 照片乱码 bug(docs/07)从链路中消失。
3. **板上原生可部署**:不用刷 JP7,当前 JP6.2 环境即可。

## 可行性依据

- 已验证 "TRT ViT → HF LLM" 双图零串扰(hybrid_trt_vit_multiimg.json)——
  证明 TRT ViT 特征能被 Qwen LLM 正确消费;llama.cpp LLM 是同一模型。

## mrope 位置布局复刻配方(✅ 已从源码解决)

源码依据:llama.cpp tools/mtmd/mtmd-helper-common.h,struct decode_embd_batch。

**布局 = [seq_len × 4] 分段块状(section-major),4 个连续块:**
```c
// set_position_mrope_2d(视觉) / set_position_mrope_1d(文本):
pos[i          ] = t   // 块0 时间
pos[i + N      ] = y   // 块1  = 行 = height   ← Qwen mrope_section 的 h
pos[i + N*2    ] = x   // 块2  = 列 = width    ← Qwen mrope_section 的 w
pos[i + N*3    ] = z   // 块3  = 0(未用)
// 文本档:四块全 = pos_0+i(标量复制)
```

**直通映射**:我们 runtime/mrope.py 的 position_ids[3,N]=(t,h,w) 已与 HF 逐元素对拍一致 →
块0=第0行(t)、块1=第1行(h)、块2=第2行(w)、块3=0。零重算,带步长拷贝即可。

**分段约束**:llama_batch 是 token XOR embd(embd 在 token=NULL 时用),不能混。
按 mtmd 做法分段 decode:文本段喂 token、视觉段喂 embd(TRT 特征),KV 跨段累积,
位置全序列一次算好按段切片。

**section 切分免管**:[16,24,24] 的 t/h/w rotary 维分配由 llama.cpp attention 读 GGUF
元数据自行处理——我们只喂 4 个位置块。

## 实现选型

- **A(推荐先做)**:llama-cpp-python 低层 API,pos/embd/token 是 ctypes 数组;
  全 Python 原型:TRT ViT(pycuda)→ numpy 特征 → llama_batch → decode。天级出原型。
- **B**:C++ 驱动链接 TensorRT + llama.cpp,性能最优,需编译。

## 剩余待验证暗礁

1. **特征数值对齐**:TRT ViT 输出维度天然对上 LLM 隐层(2048);数值约定 vs mmproj
   需实测(同源权重,已由 TRT ViT→HF LLM 零串扰间接背书)。

## 三条端到端路线对比

| 路线 | 工期 | 性质 | 优劣 |
|---|---|---|---|
| 外置 RoPE 补 0.12 | 周级 | 啃融合 kernel,重做 0.15 | 学习高、交付差 |
| 刷 JP7 + Edge-LLM | 半天 | 用官方工具 | 交付最快 |
| **TRT ViT → llama.cpp** | 天级 | 公开 API 集成,不 fork | **可部署+治照片bug+自己的集成工作** |

## 方法论小结

「我读穿了两个栈的内部:mtmd 高层不认外部 embedding,但它下面 llama.cpp 核心的
llama_batch.embd 是公开注入口。所以我能把 TRT 的快视觉和 llama.cpp 的端到端缝起来——
不改源码,纯公开 API 集成,还顺手绕开了 llama.cpp 的 GPU 照片数值 bug。这是集成架构,
不是调库。」
