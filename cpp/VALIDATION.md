# C++ 验证记录

日期：2026-09-06。验证范围为本机语法/API 编译、CPU 逻辑和预处理参考对拍。

## 编译环境

| 项目 | 实际使用值 |
|---|---|
| 平台 | Windows x64 |
| 编译器 | MSVC 19.44.35217.0，C++17，`/W4 /utf-8 /permissive-` |
| CMake | 3.31.6-msvc6；Visual Studio 17 2022 生成器 |
| 配置 | Release，`QWEN_CHECK_INFERENCE=ON` |
| CUDA 头文件 | 本机 CUDA Toolkit 11.8.89 |
| TensorRT 头文件 | 官方 10.3，`5b990f0a739d8faf962dfe54f4829942633c639c` |
| llama.cpp 头文件 | 官方 `ddd4ec1428a6201e18975ea52b07c71e0f9aef26` |
| Python 对拍环境 | Transformers 4.57.1、Pillow 12.2.0、NumPy 2.4.4 |

CUDA 11.8 在这里仅提供本机编译声明；**这不构成 TensorRT 10.3 与该 CUDA 版本的
运行兼容性结论**。全部外部头是上游真实文件，未使用伪造 SDK 或模型替身。
头文件及 vendored 单头依赖的 SHA-256 见 `dependencies.lock.json`。

## 已完成检查

1. 所有 8 个生产 C++ 翻译单元和 4 个测试翻译单元通过 MSVC 编译。
   CPU 库、`qwen_preprocess` 及测试程序成功链接。TensorRT、llama.cpp 封装和
   `vlm_cli` 通过真实 SDK 头的目标文件编译，**未执行完整推理程序链接**。
2. CTest：6/6 通过。覆盖 `preprocess`、`mrope`、`io`、`options`、
   `preprocess_help`、`preprocess_missing_args`。
3. 真实 HF/Pillow 全量张量对拍：11/11 通过。各用例观察到的最大绝对误差均为 0。
   脚本容差为 `rtol=0, atol=2e-6`；没有把未测试的图片/平台推断为逐元素一致。
4. 生产图片读取入口对拍：PNG、PPM 到 `[4096,1176]` 的完整预处理张量误差为 0。
   JPEG 通过解码、形状及有限值检查，没有声称与 Pillow JPEG 解码逐像素一致。
5. 38 个锁定上游头文件通过 SHA-256 校验；依赖下载工具的本地复核路径通过。

预处理用例：

| 输入尺寸 | 固定输出尺寸 | 最大绝对误差 |
|---|---|---|
| 84×56 | 84×56 | 0 |
| 11×17 | 28×28 | 0 |
| 97×65 | 56×28 | 0 |
| 56×19 | 56×28 | 0 |
| 29×56 | 56×56 | 0 |
| 300×17 | 28×56 | 0 |
| 1×33 | 28×28 | 0 |
| 33×1 | 28×28 | 0 |
| 1920×1080 | 896×896 | 0 |
| 51×79 | 896×896 | 0 |
| 896×896 | 896×896 | 0 |

参考口径为 Pillow BICUBIC 固定 resize 后调用 `Qwen2VLImageProcessor(do_resize=False)`。
上述测试不覆盖 HF 动态分辨率选择、视频时间编码、batch padding 或模型推理。

## 复现命令

在已加载 Visual Studio/CMake 到 PATH 的 PowerShell 中，从仓库根目录执行：

```powershell
python cpp/tools/fetch_check_headers.py
cmake -S cpp -B .build/cpp-msvc -G "Visual Studio 17 2022" -A x64 -DQWEN_CHECK_INFERENCE=ON "-DLLAMA_ROOT=$PWD/.build/cpp-deps/llama" "-DTENSORRT_ROOT=$PWD/.build/cpp-deps/tensorrt"
cmake --build .build/cpp-msvc --config Release --parallel 4
ctest --test-dir .build/cpp-msvc -C Release --output-on-failure
python cpp/tests/compare_preprocess.py --executable .build/cpp-msvc/Release/test_preprocess.exe
python cpp/tests/check_preprocess_cli.py --executable .build/cpp-msvc/Release/qwen_preprocess.exe
```

对拍命令中的 Python 必须来自装有上述参考依赖的环境。原始 CTest 日志位于
本机未跟踪目录 `.build/cpp-msvc/Testing/Temporary/LastTest.log`。

## 未验证内容

未运行 GPU、TensorRT engine、GGUF 模型或 Orin 设备；未验证完整链接、ABI、
驱动/设备兼容性、端到端答案或稳定性；没有新版本延迟、吞吐、功耗或精度结果。
原仓库 `results/verified/` 中的性能数据对应原实现，不能归给本次 C++ 重构。
