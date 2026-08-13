"""
编译：CUDA_ARCH=<SM架构号> python setup.py install
    Orin (SM87，默认值)：  python setup.py install
    桌面 4070TiS (SM89)：  CUDA_ARCH=89 python setup.py install
架构号通过环境变量 CUDA_ARCH 传入（不是命令行参数）。
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

arch = os.environ.get("CUDA_ARCH", "87")   # Orin = SM87

setup(
    name="token_merge_cuda",
    ext_modules=[
        CUDAExtension(
            name="token_merge_cuda",
            sources=["bindings.cpp", "token_merge_launcher.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}",
                         "--use_fast_math", "-lineinfo"],   # lineinfo 给 Nsight 用
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
