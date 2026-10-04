import sys, time, warnings
import torch

sys.path.insert(0, "/app/Salience-DETR")
from util.lazy_load import Config
from util.utils import load_checkpoint, load_state_dict

CONFIG_PATH = "configs/salience_detr/salience_detr_swin_l_800_1333.py"
WEIGHT_PATH  = "/app/Salience-DETR/checkpoints/salience_detr_swin_l_800_1333/train/2026-07-23-19_59_20/best_ap.pth"
DEVICE = "npu:0"
warnings.filterwarnings("ignore")


def build():
    m = Config(CONFIG_PATH).model
    try:
        ckpt = load_checkpoint(WEIGHT_PATH)
        ckpt = ckpt["model"] if "model" in ckpt else ckpt
        load_state_dict(m, ckpt)
    except Exception as e:
        print(f"[WARN] 权重加载异常: {e}")
    return m


# ---------- 1. 参数量 ----------
model = build()
params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
params_total = sum(p.numel() for p in model.parameters())
print(f"[Params] trainable: {params_train/1e6:.2f} M")
print(f"[Params] total:     {params_total/1e6:.2f} M")

# ---------- 2. FLOPs（模型含 NPU 自定义算子，只能在 NPU 上执行） ----------
model_f = build().to(DEVICE).eval()
dummy_npu = torch.randn(3, 800, 1333, device=DEVICE)
flops = None

# 方法1: thop（NPU 真实前向 + hook 计数）
try:
    from thop import profile
    macs, _ = profile(model_f, inputs=([dummy_npu],), verbose=False)
    flops = macs * 2
    print(f"[FLOPs] thop(NPU) 成功: {flops/1e9:.2f} G")
except Exception as e:
    print(f"[FLOPs] thop(NPU) 失败: {e}")

# 方法2: fvcore（NPU，train 模式规避后处理 NoneType）
if flops is None:
    try:
        from fvcore.nn import FlopCountAnalysis
        model_f.train()
        fca = FlopCountAnalysis(model_f, ([dummy_npu],))
        flops = fca.total()
        print(f"[FLOPs] fvcore(NPU) 成功: {flops/1e9:.2f} G")
    except Exception as e:
        print(f"[FLOPs] fvcore(NPU) 失败: {e}")

if flops is None:
    print("[FLOPs] NPU 上所有方法失败，跳过（仅报 params + latency）")

# ---------- 3. 延迟（NPU；重建干净模型清除 hook 残留） ----------
print(">>> 开始 NPU 延迟测试（warmup 10 + 计时 100 次），请耐心等待...")
model = build().to(DEVICE).eval()
dummy_npu = torch.randn(3, 800, 1333, device=DEVICE)

for _ in range(10):
    model([dummy_npu])
    torch.npu.synchronize()

N = 100
torch.npu.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    model([dummy_npu])
torch.npu.synchronize()
t1 = time.perf_counter()
ms = (t1 - t0) / N * 1000

print("=" * 56)
print(f"[Latency] batch=1, {DEVICE}: {ms:.2f} ms/frame")
print("=" * 56)
flop_str = f"{flops/1e9:.2f} G" if flops else "N/A"
print(f"SUMMARY  params(train)={params_train/1e6:.2f}M  "
      f"params(total)={params_total/1e6:.2f}M  "
      f"flops={flop_str}  latency={ms:.2f} ms")
