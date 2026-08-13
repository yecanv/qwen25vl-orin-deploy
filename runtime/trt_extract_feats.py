# 板侧: TRT ViT 引擎现产视觉特征 → fp32 raw bin(喂 C++ 驱动)
import numpy as np, tensorrt as trt, pycuda.driver as cuda, pycuda.autoinit, os, time, json

Z = np.load(os.path.expanduser("~/hybrid/hybrid_inputs.npz"))
lg = trt.Logger(trt.Logger.ERROR)
eng = trt.Runtime(lg).deserialize_cuda_engine(
    open(os.path.expanduser("~/vitq/vit_fp16_st.engine"), "rb").read())
c = eng.create_execution_context(); s = cuda.Stream()

class OA(trt.IOutputAllocator):
    def __init__(x): super().__init__(); x.mem, x.size, x.shape = None, 0, None
    def reallocate_output(x, n, m, sz, a):
        if sz > x.size: x.mem = cuda.mem_alloc(sz); x.size = sz
        return int(x.mem)
    def reallocate_output_async(x, n, m, sz, a, st): return x.reallocate_output(n, m, sz, a)
    def notify_shape(x, n, sh): x.shape = tuple(sh)

al = OA(); c.set_output_allocator("vision_embeds", al)
report = {}
for tag in ("a", "b"):
    pix = Z["pix_" + tag].astype(np.float16)
    grid = Z["grid_" + tag].ravel().tolist()
    try: c.set_input_shape("pixel_values", pix.shape)
    except Exception: pass
    dp = cuda.mem_alloc(pix.nbytes); cuda.memcpy_htod(dp, np.ascontiguousarray(pix))
    c.set_tensor_address("pixel_values", int(dp))
    c.execute_async_v3(s.handle); s.synchronize()          # 预热
    t0 = time.time(); c.execute_async_v3(s.handle); s.synchronize()
    vit_ms = (time.time() - t0) * 1000
    out = np.empty(al.shape, dtype=np.float16); cuda.memcpy_dtoh(out, al.mem)
    out.astype(np.float32).tofile(os.path.expanduser(f"~/hybrid/feat_{tag}.bin"))
    report[tag] = {"grid_thw": grid, "out_shape": list(al.shape), "vit_ms": round(vit_ms, 1)}
    print(tag, report[tag])
json.dump(report, open(os.path.expanduser("~/hybrid/vit_report.json"), "w"))
print("EXTRACT_OK")
