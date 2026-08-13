# 环境搭建与踩坑排查

> 这份文档按**报错信息**索引。跑不通时 Ctrl+F 搜报错关键词。
> 遇到没收录的问题，群里发我完整日志。

---

## 一、开始之前的三个确认

```bash
# 1. 板卡型号（决定跑 3B 还是 2B）
cat /proc/device-tree/model

# 2. JetPack 版本（决定 TRT-LLM 用哪个 tag）
cat /etc/nv_tegra_release
dpkg-query -W -f='${Version}' nvidia-jetpack

# 3. TensorRT 版本（版本对不上编译必失败）
dpkg -l | grep libnvinfer
```

把这三个结果发我，我确认 TRT-LLM 该用哪个 tag 再开始编译。
**不要跳过这步直接编**——版本对不上，四小时白费。

---

## 二、内存相关（8GB 板子的主要战场）

### 症状：`trtllm-build` 到一半进程被 kill，没有报错信息

被 OOM killer 干掉了。确认：

```bash
dmesg | grep -i "killed process"
```

解法（按顺序做）：

1. **开 swap**（`env/jetson_setup.sh` 会做）
   ```bash
   free -h    # swap 至少 16G
   ```
2. **关图形界面**，省 600MB~1GB
   ```bash
   sudo systemctl set-default multi-user.target && sudo reboot
   ```
3. **关掉 zram**。Jetson 默认开 zram，会和真 swap 抢 CPU
   ```bash
   sudo systemctl disable nvzramconfig && sudo reboot
   ```
4. **降低 build 并行度**：`--job_count 2`
5. 还不行就**换到 16GB 板子**，或者退到 Qwen2-VL-2B

### 症状：运行时 OOM，但 build 时没问题

`kv_cache_free_gpu_memory_fraction` 设太高了。Orin 是统一内存，
这个 fraction 是相对于**全部系统内存**，不是独立显存。

8GB 板子建议 0.4~0.5，16GB 建议 0.55~0.65。留够给：
- ViT engine（约 1.5GB）
- 系统本身（约 1.2GB）
- 图像预处理的临时缓冲

### 症状：跑起来极慢，比预期慢 10 倍以上

在走 swap。检查：

```bash
free -h        # Swap used 不为 0 就是了
tegrastats     # 看 SWAP 字段
```

swap 只能救 build，**不能救运行时**。运行时一走 swap 就是灾难。
解法是减小 max_batch_size / max_seq_len / max_multimodal_len。

---

## 三、Engine 相关

### 症状：`deserializeCudaEngine` 返回 nullptr / engine 加载失败

**最常见的原因：engine 不是在这台板子上 build 的。**

TensorRT engine 绑定：
- SM 架构（Orin 是 sm_87）
- TensorRT 版本（大版本不兼容，小版本也常常不兼容）
- cuDNN / cuBLAS 版本

在桌面 4070Ti（sm_89）上编的 engine，Orin 上百分之百加载不了。
**必须在目标板卡上重新 build。** 这不是配置问题，是设计如此。

### 症状：`Input shape out of bounds` / shape mismatch

optimization profile 没覆盖到实际输入。

```
n_patches = (H/14) × (W/14) × 2
```

比如 1280×720 的图 → (1280/14)×(720/14)×2 ≈ 91×51×2 = 9282 patches，
超过默认 MAX_PATCHES=16384 没问题，但如果你调小了就会炸。

解法：
```bash
MAX_PATCHES=32768 bash convert/build_vit_engine.sh
```
或者在 processor 里限制 `max_pixels`。

### 症状：`max_multimodal_len` 相关报错

build LLM engine 时的 `--max_multimodal_len` 必须 ≥ 单次请求的最大视觉 token 数：

```
max_multimodal_len ≥ MAX_PATCHES / 4
```

（除以 4 是因为 patch merger 做 2×2 合并）

注意这个值会占 KV Cache 预算，设太大会导致可用 batch 变小。

---

## 四、输出不对（最难查的一类）

### 症状：输出乱码 / 完全无意义的字符

大概率是 **M-RoPE position_ids 算错**。

先跑自检：
```bash
python3 runtime/mrope.py     # 会跟 HF 原版对拍
```

不通过就别往下做，后面所有数据都不可信。

### 症状：能说人话，但"看图说瞎话"——描述的内容和图片无关

按可能性排序：

1. **prompt table 没接上**。检查占位符数量是否等于视觉 token 数：
   ```python
   assert (input_ids == 151655).sum() == vision_embeds.shape[0]
   ```
2. **fake id 算错**。必须是 `vocab_size + row_index`，
   注意 `vocab_size` 要用 `len(tokenizer)` 而不是 config 里的
   （两者常常不等，config 的 vocab_size 通常是 padded 后的值）
3. **ViT engine 输出维度不对**。检查 merger 输出的 hidden 是否等于 LLM 的 hidden_size

### 症状：前面正常，输出十几个 token 后开始重复 / 语无伦次

**decode 阶段忘了加 `mrope_delta`。**

见 `runtime/mrope.py` 的 `decode_step_position()`。
prefill 结束时保存的 delta，decode 每一步都要加上。

### 症状：文字问答正常，一涉及看图就答错

**校准集问题**——用纯文本校准了量化。

VLM 的 LLM 主干在推理时输入里有大量视觉 token，其激活分布和文本完全不同。
纯文本校准出来的 scale 覆盖不住视觉 token 的动态范围。

解法：用 `calib/build_vl_calib.py` 生成的图文混合校准集重新量化。

**这个现象本身就是很好的面试素材**——你能讲清楚"为什么 VLM 校准必须走多模态
前向"，而且有对照实验数据，比背概念强得多。

### 症状：OCR / 图中文字识别明显变差

ViT 被量化了。检查 `quant_recipe.json` 里的 `excluded_modules`
是否包含所有视觉模块。

`convert/quantize_llm.py` 里的 `VISION_MODULE_PATTERNS` 是按 Qwen2.5-VL 的
命名写的，如果你换了模型，模块名可能不同，先打印一遍：

```python
for n, _ in model.named_modules():
    print(n)
```

---

## 五、性能不符合预期

### 症状：延迟比预期高很多

检查清单（按影响大小排序）：

```bash
sudo nvpmodel -q          # 是否 MAXN？非 MAXN 性能能差 40%
sudo jetson_clocks --show # 频率是否锁定？重启后会失效
tegrastats                # 结温多少？>85°C 已经在降频了
free -h                   # 是否在走 swap？
```

**`jetson_clocks` 每次重启后都要重新执行。** 这是最常见的"昨天还好好的今天变慢了"。

### 症状：多次测量结果波动很大

1. 没预热。前几次跑包含 kernel autotuning 和内存分配，必须丢掉。
   本仓所有 benchmark 脚本默认预热 3~5 次。
2. 没控温。上一轮测试的余热会影响下一轮，`run_all.sh` 里每轮之间 sleep 120s。
3. 后台有其他进程。测之前 `htop` 看一眼。

### 症状：并发数上去了但吞吐没涨

VLM 的并发瓶颈通常不是 KV Cache，而是 `max_multimodal_len × batch`
超了预算。检查 build 参数，或者降低输入分辨率。

---

## 六、TRT-LLM 编译

### 版本匹配

这是最大的坑。TRT-LLM 的每个 tag 只支持特定的 TensorRT 版本，
JetPack 又绑定了 TensorRT 版本。三者必须对上。

**做法**：先查你的 JetPack 带的 TensorRT 版本，再去 TRT-LLM 的
release notes 找对应 tag。查不到就发我，我帮你确认。

**Qwen2-VL 支持从 v0.12 起；Qwen2.5-VL 需要更新的 tag，务必先确认。**
如果你的 JetPack 版本只能配到不支持 Qwen2.5-VL 的 tag，
就退到 Qwen2-VL-2B——架构同源，本仓代码用 `--model-family` 切换即可。

### 编译加速

```bash
--cuda_architectures "87-real"    # 只编 SM87，省一半时间
--job_count 4                     # 8GB 板子用 2
```

### 兜底方案

如果 TRT-LLM 在你的 JetPack 版本上实在编不出来，还有两条路：

- **ViT 走 TensorRT + LLM 走 llama.cpp(混合链路,已实测交付)**:手写约 170 行 C++ 驱动(runtime/hybrid_driver.cpp)零改 llama.cpp 源码,经公开 API llama_batch.embd 注入 TensorRT ViT 特征。板上实测 TTFT≈1.88s(未含图像预处理与特征IO)对比纯 llama.cpp 链 P50 3.59s(含预处理)提速约 1.7~1.9 倍,decode 吞吐与纯链一致。数据:results/verified/orin/return_day2/hybrid_llamacpp_e2e.json。
- **MLC-LLM**：对 Jetson 支持较好，量化方案是 q4f16。

这条路不是兜底而是本项目的主交付:定位 TRT-LLM 0.12 的 M-RoPE 兼容边界后,手写 C++ 驱动将 TensorRT ViT 特征注入 llama.cpp,混合链路使 VLM 首字延迟由 3.59 s 降至约 2 s(注意口径:1.88s 未含预处理,3.59s 含预处理)。

---

## 七、遇到没收录的问题

发我这些信息，别只发一句"跑不了"：

1. 完整报错栈（不是截图最后一行）
2. `python3 benchmark/probe_device.py` 的输出
3. 对应的 build log（`logs/` 目录下）
4. 你执行的完整命令

有这四样，大部分问题我能直接定位。
