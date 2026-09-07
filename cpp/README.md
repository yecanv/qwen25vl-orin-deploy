# C++17 单图推理运行时

本目录把已有的“Python TensorRT 特征提取 + C++ LLM 驱动”整理为独立 C++
运行时，提供图像读取、固定尺寸预处理、TensorRT 视觉推理、M-RoPE、llama.cpp
解码、配置和 JSON 计时。离线导出、量化、评测和旧基线继续保留在原目录。

**验证状态：本机 C++ 编译检查及 CPU 测试通过；没有验证完整推理程序的链接、
GPU 执行、Orin 兼容性、生成质量或性能。旧版本的板端数字不能作为本版本结果。**
具体环境和检查见 [VALIDATION.md](VALIDATION.md)。

## 范围与输入契约

```text
JPEG / PNG / PPM + 问题
  → RGB / 固定尺寸 BICUBIC / normalize / patchify
  → TensorRT ViT → FP32 主存特征 → 四块 M-RoPE → llama.cpp → 流式文本
```

- 单请求、单张静态图片、greedy 解码；没有多图、视频、并发服务、视觉 token 压缩。
- 默认使用 `onnx/vit_fp16_static.shapes.json`：896×896，grid `[1,64,64]`，
  输入 `[4096,1176]`，视觉输出 `[1024,2048]`。必须使用与 engine 对应的导出契约；
  仅凭 token 数不能区分具有相同面积的不同网格。
- 输入图像强制缩放到契约尺寸，**可能改变宽高比**；这不是 HF 动态 `smart_resize`。
  RGB 预处理对拍是“Pillow 固定 resize + HF `do_resize=False`”口径。
- 图片解码不自动应用 EXIF 旋转；JPEG 解码器与 Pillow 可能产生像素级差异。
- TensorRT 输入必须是静态二维 `pixel_values`；支持 FP16/FP32、DEVICE/LINEAR I/O。
  若输出保守标记为 `-1`，用受限 `IOutputAllocator` 分配并核对实际形状。
- 特征直接经内存交接，仍有设备到主存及后续上传，**没有实现 GPU 零拷贝**。
- 模型加载一次；每次 `generate` 创建独立 context。类实例不接受重叠调用。

## 模块

| 文件 | 职责 |
|---|---|
| `src/preprocess.cpp` | RGB 缩放、CLIP 归一化、14/2/2 patch 排列 |
| `src/mrope.cpp` | 单图四块位置与图后文本起点 |
| `src/io.cpp` | 图片解码、静态形状契约检查、CPU 张量导出 |
| `src/options.cpp` | JSON 配置、命令行覆盖、参数验证 |
| `src/vit_engine.cpp` | TensorRT/CUDA RAII、缓冲复用与执行检查 |
| `src/llama_decoder.cpp` | GGUF、tokenizer、容量检查、上下文与流式生成 |
| `apps/preprocess_cli.cpp` | 无 GPU 的真实图片预处理入口 |
| `apps/vlm_cli.cpp` | 完整单图混合链路入口源码 |

## 无 GPU 构建与测试

需要 CMake ≥ 3.20 和 C++17 编译器。默认不查找 TensorRT、CUDA 或 llama.cpp，
不联网下载依赖；图像和 JSON 解析的单头文件已保留在 `third_party/`。
以下命令从仓库根目录执行：

```sh
cmake -S cpp -B .build/cpp-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build .build/cpp-cpu --config Release --parallel
ctest --test-dir .build/cpp-cpu -C Release --output-on-failure
```

Visual Studio 多配置生成器的程序在 `.build/cpp-cpu/Release/`；Linux 单配置生成器
的程序通常直接在 `.build/cpp-cpu/`。

实际图片预处理（Linux 路径示例）：

```sh
.build/cpp-cpu/qwen_preprocess --image assets/demo896.jpg \
  --vision-contract onnx/vit_fp16_static.shapes.json --output .build/pixels.f32
```

输出为主机原生 float32 的 row-major `pixel_values`，不是视觉 embedding。
可选的全量对拍不加载模型、不访问网络，但需要本地 NumPy、Pillow、Transformers：

```sh
python cpp/tests/compare_preprocess.py --executable .build/cpp-cpu/test_preprocess
python cpp/tests/check_preprocess_cli.py --executable .build/cpp-cpu/qwen_preprocess
```

## 推理源码编译检查：不链接、不运行 GPU

此模式使用真实 SDK 头编译目标文件，能够发现语法、类型和声明层面的 API 不匹配，
不能验证库 ABI、链接符号、engine 加载或模型输出。需要已安装 CUDA Toolkit 的头文件；
不需要 GPU、engine 或 GGUF。

```sh
python cpp/tools/fetch_check_headers.py
cmake -S cpp -B .build/cpp-check -DQWEN_CHECK_INFERENCE=ON \
  -DLLAMA_ROOT="$PWD/.build/cpp-deps/llama" \
  -DTENSORRT_ROOT="$PWD/.build/cpp-deps/tensorrt"
cmake --build .build/cpp-check --config Release --target check_inference --parallel
```

PowerShell 的 `$PWD` 同样可用于上述路径，但续行请写成单行或使用 PowerShell 续行语法。
CUDA 未被自动找到时添加 `-DCUDAToolkit_ROOT=实际CUDA安装目录`。

头文件下载脚本只取 `dependencies.lock.json` 中固定提交的文件并校验 SHA-256：
llama.cpp `ddd4ec1428a6201e18975ea52b07c71e0f9aef26`，TensorRT 10.3
`5b990f0a739d8faf962dfe54f4829942633c639c`。

## 完整推理构建说明（本次未验证链接/执行）

准备匹配目标设备的 TensorRT/CUDA SDK，以及上述固定版本 llama.cpp 的**共享库**构建。
例如在已有 llama.cpp 源码目录中，启用 `-DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON`。
本项目通过外部库路径链接，不自动修改或重新编译 llama.cpp。

```sh
cmake -S cpp -B .build/cpp-runtime -DCMAKE_BUILD_TYPE=Release \
  -DQWEN_BUILD_INFERENCE=ON \
  -DLLAMA_ROOT=/path/to/llama.cpp -DTENSORRT_ROOT=/path/to/TensorRT
cmake --build .build/cpp-runtime --parallel
```

系统安装的 TensorRT 也可自动发现；必要时显式设置 `TENSORRT_INCLUDE_DIR`、
`TENSORRT_LIBRARY`、`LLAMA_INCLUDE_DIR`、`GGML_INCLUDE_DIR`、`LLAMA_LIBRARY`、`GGML_LIBRARY`。
部署时须让 `libllama`、`libggml`、`libggml-base` 及 CPU/CUDA 后端共享库可被加载器发现。
这里不支持用裸静态库替代这些共享库。

编辑 `cpp/configs/single_image.json` 中的模型、engine 和图片路径，然后：

```sh
.build/cpp-runtime/vlm_cli --config cpp/configs/single_image.json \
  --prompt "请描述这张图片。" --max-new-tokens 128 --metrics .build/inference.json
```

配置内和命令行内的文件路径都相对于**当前工作目录**。命令行值覆盖配置值，
未知参数、非法整数和缺失必填项会报错。`--max-new-tokens 0` 仅执行 prefill。
输出文本写 stdout，诊断与 `TIMING_JSON` 写 stderr。

## 计时与正确性边界

- `load_ms`：ViT/LLM 加载；与请求耗时分列。
- `preprocess_ms`：读取图片、解码、缩放、归一化和 patchify。
- `vision_ms`：输入转换/上传、ViT、输出回传/转换；不是纯 GPU kernel 时间。
- `llm_first_token_ms`：进入 `generate` 到首个非 EOG token 的采样，含 context 创建及 prefill。
- `request_first_output_ms`：开始读图片到首次非空文本写出并 flush；无文本时为 null。
- `request_ms`：整个请求时间，含所有输出与 context 收尾。

首次运行没有隐式预热，不能直接拿一次结果与历史 P50 对比。本次交付只确认文档列出的
编译和 CPU 检查，不产生新的 TTFT、吞吐或模型精度结论。
