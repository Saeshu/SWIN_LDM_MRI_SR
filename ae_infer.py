import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ----------------------------
# 1. Paths (EDIT THIS)
# ----------------------------
AE_CKPT_PATH = "/workspace/checkpoints/ae_patch_swin_step8000.pt"  # <- change if needed

# ----------------------------
# 2. Load models
# ----------------------------
enc = ShapedEncoder3D().cuda()
dec = Decoder3D().cuda()

ckpt = torch.load(AE_CKPT_PATH, map_location="cuda")

enc.load_state_dict(ckpt["encoder"], strict=True)
dec.load_state_dict(ckpt["decoder"], strict=True)

enc.eval()
dec.eval()

print("✅ AE checkpoint loaded")

# ----------------------------
# 3. Grab a sample
# ----------------------------
vol = dataset[0]  # [1, D, H, W]

x = random_patch_3d(vol, 64).unsqueeze(0).cuda()

# ----------------------------
# 4. Forward pass
# ----------------------------
with torch.no_grad():
    z, skip = enc(x)
    x_hat = dec(z, skip)
    x_hat = match_shape(x_hat, x)

# ----------------------------
# 5. Print quick stats
# ----------------------------
print("Input mean:", x.mean().item())
print("Recon mean:", x_hat.mean().item())
print("L1 loss:", F.l1_loss(x_hat, x).item())

# ----------------------------
# 6. Visualize center slice
# ----------------------------
mid = x.shape[2] // 2

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(x[0,0,mid].cpu(), cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Reconstruction")
plt.imshow(x_hat[0,0,mid].cpu(), cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
