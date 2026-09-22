"""Reports which GPUs PyTorch can see, and which one laya-server will use.

ROCm exposes AMD GPUs through the `torch.cuda` API, so "cuda" below means a
Radeon as well as an NVIDIA card.
"""

import torch

from server import default_device

print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, HIP {torch.version.hip}")

if not torch.cuda.is_available():
    print("No GPU visible to PyTorch; Laya will run on the CPU.")
    raise SystemExit(0)

for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    arch = getattr(props, "gcnArchName", "")
    kind = "integrated" if getattr(props, "is_integrated", 0) else "discrete"
    print(f"  cuda:{index}  {props.name:<28} {arch:<8} {kind}")

device = default_device()
if device is None:
    print("No discrete GPU found; Laya will pick its own device. Pass --device to choose.")
else:
    print(f"laya-server will use {device} ({torch.cuda.get_device_name(torch.device(device))}).")
