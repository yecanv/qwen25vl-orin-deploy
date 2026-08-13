@echo off
rem =========================================================================
rem Windows desktop build: CPU ref + CUDA extension (RTX 4070TiS, SM89)
rem Usage: build_win.bat   (paths resolved via %~dp0, safe with CJK dirs)
rem Env pairing (do NOT change casually):
rem   - conda env "yolo" = torch 2.2.2+cu118  <-> system nvcc 11.8
rem     (qwen_trt's torch is cu130: major-version mismatch with nvcc, rejected
rem      by torch cpp_extension version check)
rem   - NVCC_APPEND_FLAGS=-allow-unsupported-compiler: CUDA 11.8 officially
rem     supports MSVC up to VS2022 17.3; newer toolsets trip nvcc's guard.
rem     The flag skips the guard; if the build then fails, install the
rem     v143 17.3 toolset instead.
rem =========================================================================
call "D:\APP\AboutCode\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
cd /d "%~dp0"

echo === [1/3] compile CPU reference (cl /O2) ===
if not exist ref\NUL mkdir ref
cl /nologo /O2 /EHsc /utf-8 ref\token_merge_ref.cpp /Fe:ref\token_merge_ref.exe /Fo:ref\
if errorlevel 1 exit /b 2

echo === [2/3] run CPU reference: 8 boundary cases (fused-mirror vs naive) ===
ref\token_merge_ref.exe
if errorlevel 1 exit /b 3

echo === [3/3] build CUDA extension (SM89, torch cu118 env "yolo") ===
set CUDA_ARCH=89
rem Two independent version gates when pairing CUDA 11.8 with new MSVC:
rem   gate 1: nvcc's host-compiler check      -> -allow-unsupported-compiler
rem   gate 2: MSVC STL's yvals_core.h STL1002 -> -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
rem If runtime misbehaves under this bypass, the clean fix is a torch-cu12x
rem env paired with the installed CUDA 12.8 toolkit (>=12.4 passes gate 2).
set NVCC_APPEND_FLAGS=-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
set PYTHONIOENCODING=utf-8
set DISTUTILS_USE_SDK=1
call conda run -n yolo --no-capture-output python setup.py build_ext --inplace
if errorlevel 1 exit /b 4

echo === DONE ===
