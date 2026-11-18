import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path("C:/Users/Varun's LOQ/Desktop/GlacierResearch")
sys.path.insert(0, str(ROOT))

from src.unet_baseline import GlacierUNet
from src.fewshot_model import FewShotProtoNet
from src.dataset import DL4GAMDataset

print("Checking PyTorch & CUDA:")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 1. Test U-Net
unet_path = ROOT / "outputs/runs/objective1_run_002_recovery/checkpoints/unet_baseline_best.pt"
print(f"Loading U-Net from {unet_path}...")
unet = GlacierUNet(in_channels=13, base_channels=16).to(device)
ckpt = torch.load(unet_path, map_location=device, weights_only=False)
state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
unet.load_state_dict(state_dict)
unet.eval()
print("U-Net loaded successfully!")

# 2. Test ProtoNet
proto_path = ROOT / "outputs/runs/objective1_run_002_recovery/checkpoints/fewshot_protonet_best.pt"
print(f"Loading ProtoNet from {proto_path}...")
proto = FewShotProtoNet(in_channels=13, embed_dim=64, temperature=20.0).to(device)
ckpt_p = torch.load(proto_path, map_location=device, weights_only=False)
state_dict_p = ckpt_p["model_state_dict"] if "model_state_dict" in ckpt_p else ckpt_p
proto.load_state_dict(state_dict_p)
proto.eval()
print("ProtoNet loaded successfully!")

# 3. Test NetCDF read on a sample
ds_train = DL4GAMDataset(root=ROOT, fold="fold_train")
print(f"Train patches: {len(ds_train)}")
ds_test = DL4GAMDataset(root=ROOT, fold="fold_test")
print(f"Test patches: {len(ds_test)}")

gid, sample_path = ds_test.patch_list[0]
print(f"Sample test patch: {sample_path}")
bands, mask, meta = DL4GAMDataset.read_netcdf_patch(sample_path)
print(f"Bands shape: {bands.shape}, Mask shape: {mask.shape}, Ratio: {meta['glacier_pixel_ratio']:.3f}")

# Run U-Net forward
x_norm = (bands - ds_train.mean) / (ds_train.std + 1e-8)
x_tensor = torch.from_numpy(x_norm).unsqueeze(0).float().to(device)

t0 = time.perf_counter()
with torch.no_grad():
    out_unet = unet(x_tensor)
    prob_unet = torch.sigmoid(out_unet).squeeze().cpu().numpy()
lat_unet = (time.perf_counter() - t0) * 1000
print(f"U-Net inference done in {lat_unet:.2f} ms! Output shape: {prob_unet.shape}, Range: [{prob_unet.min():.3f}, {prob_unet.max():.3f}]")

# Prepare support set for ProtoNet (e.g. K=5 from train fold)
support_x_list = []
support_y_list = []
for i in range(5):
    _, p_path = ds_train.patch_list[i]
    b, m, _ = DL4GAMDataset.read_netcdf_patch(p_path)
    b_norm = (b - ds_train.mean) / (ds_train.std + 1e-8)
    support_x_list.append(torch.from_numpy(b_norm).float())
    support_y_list.append(torch.from_numpy(m).float().unsqueeze(0))

support_x = torch.stack(support_x_list).to(device)
support_y = torch.stack(support_y_list).to(device)

t0 = time.perf_counter()
with torch.no_grad():
    out_proto = proto(support_x, support_y, x_tensor)
    prob_proto = out_proto["probs"].squeeze().cpu().numpy()
lat_proto = (time.perf_counter() - t0) * 1000
print(f"ProtoNet (K=5) inference done in {lat_proto:.2f} ms! Output shape: {prob_proto.shape}, Range: [{prob_proto.min():.3f}, {prob_proto.max():.3f}]")

print("All tests passed successfully!")
