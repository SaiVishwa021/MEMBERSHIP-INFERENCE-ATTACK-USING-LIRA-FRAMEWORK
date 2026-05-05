import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import requests

from pathlib import Path
from scipy.stats import norm as scipy_norm
from torch.utils.data import Dataset
from torchvision.models import resnet18
import torchvision.transforms as transforms


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

# Update these paths to your files.

PUB_PATH   = "/content/drive/MyDrive/tml_mia/pub.pt"
PRIV_PATH  = "/content/drive/MyDrive/tml_mia/priv.pt"
MODEL_PATH = "/content/drive/MyDrive/tml_mia/model.pt"
OUTPUT_CSV = Path("/content/drive/MyDrive/tml_mia/submission.csv")
SHADOW_DIR = Path("/content/drive/MyDrive/tml_mia/shadow_models2")

BASE_URL = "http://34.63.153.158"
API_KEY  = "...REPLACE WITH YOUR API KEY..." 
TASK_ID  = "01-mia"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 512
NUM_CLASSES = 9

# ── LiRA hyperparameters ──────────────────────
N_SHADOW      = 64    
SHADOW_EPOCHS = 100   
SHADOW_LR     = 0.1
SHADOW_WD     = 0.0   
SHADOW_BS     = 512
IN_RATIO      = 0.5

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ──────────────────────────────────────────────
# Dataset classes
# ──────────────────────────────────────────────
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids, self.imgs, self.labels = [], [], []
        self.transform = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        return self.ids[index], img, self.labels[index]

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]


# ──────────────────────────────────────────────
# Load datasets
# ──────────────────────────────────────────────
print("Loading datasets...")
pub_ds  = torch.load(PUB_PATH,  weights_only=False)
priv_ds = torch.load(PRIV_PATH, weights_only=False)

MEAN = [0.7406, 0.5331, 0.7059]
STD  = [0.1491, 0.1864, 0.1301]

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Normalize(mean=MEAN, std=STD),
])
pub_ds.transform  = transform
priv_ds.transform = transform
pub_membership = np.array(pub_ds.membership)
pub_labels_np  = np.array([pub_ds[i][2] for i in range(len(pub_ds))])

# ── Pre-extract tensors (CPU, contiguous for fast slicing) ──
print("Pre-extracting pub images...")
pub_imgs_list, pub_labels_list = [], []
for i in range(len(pub_ds)):
    item = pub_ds[i]
    pub_imgs_list.append(item[1])
    pub_labels_list.append(item[2])
pub_imgs_tensor   = torch.stack(pub_imgs_list).contiguous()
pub_labels_tensor = torch.tensor(pub_labels_list, dtype=torch.long)
N_PUB = len(pub_imgs_tensor)

print("Pre-extracting priv images...")
priv_ids_list, priv_imgs_list, priv_labels_list = [], [], []
for i in range(len(priv_ds)):
    item = priv_ds[i]
    priv_ids_list.append(item[0])
    priv_imgs_list.append(item[1])
    priv_labels_list.append(item[2])
priv_imgs_tensor   = torch.stack(priv_imgs_list).contiguous()
priv_labels_tensor = torch.tensor(priv_labels_list, dtype=torch.long)
priv_labels_np     = np.array(priv_labels_list)
N_PRIV = len(priv_imgs_tensor)


# ──────────────────────────────────────────────
# Load target model
# ──────────────────────────────────────────────
print("Loading target model...")
target_model = resnet18(weights=None)
target_model.conv1   = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
target_model.maxpool = nn.Identity()
target_model.fc      = nn.Linear(512, NUM_CLASSES)
target_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
target_model.eval().to(DEVICE)


# ──────────────────────────────────────────────
# Metric
# ──────────────────────────────────────────────
def tpr_at_fpr(scores, y, target_fpr=0.05):
    scores, y = np.asarray(scores, dtype=float), np.asarray(y)
    idx      = np.argsort(scores)[::-1]
    sorted_y = y[idx]
    neg      = (sorted_y == 0)
    fpr      = np.cumsum(neg) / (neg.sum() + 1e-12)
    cutoff   = np.searchsorted(fpr, target_fpr)
    return 0.0 if cutoff == 0 else float((sorted_y[:cutoff] == 1).mean())


# ──────────────────────────────────────────────
# Fast GPU inference — returns (N, C) logits
# ──────────────────────────────────────────────
@torch.no_grad()
def get_logits(m, imgs_tensor):
    out = []
    for start in range(0, len(imgs_tensor), BATCH_SIZE):
        imgs = imgs_tensor[start:start + BATCH_SIZE].to(DEVICE, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
            out.append(m(imgs).cpu().float())
    return torch.cat(out)  # (N, C)


def logits_to_signals(logits, labels_tensor):
    """Convert (N,C) logits + (N,) labels → dict of (N,) signals."""
    lp    = F.log_softmax(logits, dim=1)
    probs = lp.exp()
    idx   = torch.arange(len(labels_tensor))

    neg_xent = lp[idx, labels_tensor]

    logits_excl = logits.clone()
    logits_excl[idx, labels_tensor] = -1e9
    margin = logits[idx, labels_tensor] - logits_excl.max(1).values

    neg_entr = (probs * lp).sum(1)   # sum(p log p) = -H, so higher = more confident

    return {
        "neg_xent": neg_xent.numpy(),
        "margin":   margin.numpy(),
        "neg_entr": neg_entr.numpy(),
    }


# ──────────────────────────────────────────────
# Shadow model training — GPU-resident, no DataLoader
# ──────────────────────────────────────────────
def make_model():
    m = resnet18(weights=None)
    m.conv1   = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(512, NUM_CLASSES)
    return m


def train_shadow(in_idx):
    """
    Train on the IN subset with NO regularization / augmentation.
    Data stays on GPU the whole time — fastest possible training.
    """
    imgs_gpu   = pub_imgs_tensor[in_idx].to(DEVICE)
    labels_gpu = pub_labels_tensor[in_idx].to(DEVICE)
    N_train    = len(in_idx)

    m      = make_model().to(DEVICE)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == "cuda"))
    opt    = torch.optim.SGD(m.parameters(), lr=SHADOW_LR,
                             momentum=0.9, weight_decay=SHADOW_WD,
                             nesterov=True)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SHADOW_EPOCHS)

    m.train()
    for epoch in range(SHADOW_EPOCHS):
        perm       = torch.randperm(N_train, device=DEVICE)
        epoch_loss = 0.0
        n_batches  = 0
        for start in range(0, N_train, SHADOW_BS):
            b_idx  = perm[start:start + SHADOW_BS]
            imgs   = imgs_gpu[b_idx]
            labels = labels_gpu[b_idx]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
                loss = F.cross_entropy(m(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            epoch_loss += loss.item()
            n_batches  += 1
        sched.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        # Early stop once fully memorized
        if epoch >= 30 and avg_loss < 0.005:
            break

    m.eval()
    return m


# ──────────────────────────────────────────────
# Vectorized LiRA scoring (no Python loop over samples)
# ──────────────────────────────────────────────
def vectorized_lira(target_vals, shadow_vals, in_mask):
    """
    target_vals : (N,)
    shadow_vals : (N, S)
    in_mask     : (N, S) bool — True if sample i was IN shadow model s
    Returns LLR scores (N,).
    """
    N, S = shadow_vals.shape

    # Count IN/OUT per sample
    n_in  = in_mask.sum(axis=1)   # (N,)
    n_out = S - n_in

    # Masked mean/std: set OUT positions to nan for IN stats and vice versa
    vals_in  = np.where(in_mask,  shadow_vals, np.nan)
    vals_out = np.where(~in_mask, shadow_vals, np.nan)

    mu_in  = np.nanmean(vals_in,  axis=1)   # (N,)
    mu_out = np.nanmean(vals_out, axis=1)

    std_in  = np.nanstd(vals_in,  axis=1)  + 1e-8
    std_out = np.nanstd(vals_out, axis=1)  + 1e-8

    # Fixed-variance fallback for samples with too few IN/OUT runs
    global_std_in  = np.nanstd(vals_in)
    global_std_out = np.nanstd(vals_out)
    std_in  = np.where(n_in  >= 4, std_in,  global_std_in)
    std_out = np.where(n_out >= 4, std_out, global_std_out)

    llr = (scipy_norm.logpdf(target_vals, mu_in,  std_in) -
           scipy_norm.logpdf(target_vals, mu_out, std_out))
    return llr


def norm01(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-12)


# ──────────────────────────────────────────────
# Step 1: Target model signals
# ──────────────────────────────────────────────
print("\n[1/4] Target model signals...")
target_logits_pub  = get_logits(target_model, pub_imgs_tensor)
target_logits_priv = get_logits(target_model, priv_imgs_tensor)
target_pub  = logits_to_signals(target_logits_pub,  pub_labels_tensor)
target_priv = logits_to_signals(target_logits_priv, priv_labels_tensor)

print("      Signal gaps (member - non-member) on pub:")
for sig in ["neg_xent", "margin", "neg_entr"]:
    mem    = target_pub[sig][pub_membership == 1].mean()
    nonmem = target_pub[sig][pub_membership == 0].mean()
    print(f"        {sig:10s}  member={mem:.4f}  non-member={nonmem:.4f}  gap={mem-nonmem:+.4f}")


# ──────────────────────────────────────────────
# Step 2: Train shadow models, collect signals
# ──────────────────────────────────────────────
SHADOW_DIR.mkdir(parents=True, exist_ok=True)
SIG_KEYS = ["neg_xent", "margin", "neg_entr"]

shadow_pub_sigs  = {k: np.full((N_PUB,  N_SHADOW), np.nan, dtype=np.float32) for k in SIG_KEYS}
shadow_priv_sigs = {k: np.full((N_PRIV, N_SHADOW), np.nan, dtype=np.float32) for k in SIG_KEYS}
shadow_in_mask   = np.zeros((N_PUB, N_SHADOW), dtype=bool)

print(f"\n[2/4] Training {N_SHADOW} shadow models "
      f"({SHADOW_EPOCHS} epochs max, WD=0, no augmentation, device={DEVICE})...")

for s in range(N_SHADOW):
    ckpt    = SHADOW_DIR / f"shadow_{s:03d}.pt"
    rng     = np.random.RandomState(seed=s)
    in_mask = rng.rand(N_PUB) < IN_RATIO
    in_idx  = np.where(in_mask)[0]
    shadow_in_mask[:, s] = in_mask

    if ckpt.exists():
        print(f"  [{s+1:3d}/{N_SHADOW}] cached", flush=True)
        m = make_model()
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        m.eval().to(DEVICE)
    else:
        print(f"  [{s+1:3d}/{N_SHADOW}] training {len(in_idx)} samples...",
              end="", flush=True)
        m = train_shadow(in_idx)
        torch.save(m.state_dict(), ckpt)
        print(" done", flush=True)

    # Score pub + priv with this shadow model
    sp  = logits_to_signals(get_logits(m, pub_imgs_tensor),  pub_labels_tensor)
    spr = logits_to_signals(get_logits(m, priv_imgs_tensor), priv_labels_tensor)
    for k in SIG_KEYS:
        shadow_pub_sigs[k][:,  s] = sp[k]
        shadow_priv_sigs[k][:, s] = spr[k]

    del m
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# ── Shadow separation diagnostics ──
print("\n  Shadow model memorization check:")
for k in SIG_KEYS:
    in_v  = shadow_pub_sigs[k][shadow_in_mask]
    out_v = shadow_pub_sigs[k][~shadow_in_mask]
    print(f"    {k:10s}: IN={in_v.mean():.4f}  OUT={out_v.mean():.4f}  "
          f"gap={in_v.mean()-out_v.mean():+.4f}  "
          f"(need gap >> 0 for LiRA to work)")


# ──────────────────────────────────────────────
# Step 3: LiRA scoring (vectorized)
# ──────────────────────────────────────────────
#print("\n[3/4] Computing LiRA scores...")

# ── Pub: genuine IN/OUT per shadow model ──
pub_llr = {}
for k in SIG_KEYS:
    pub_llr[k] = vectorized_lira(
        target_pub[k],
        shadow_pub_sigs[k],
        shadow_in_mask,
    )

for k in SIG_KEYS:
    tpr = tpr_at_fpr(pub_llr[k], pub_membership)
    #print(f"    Pub TPR@5%FPR [{k:10s}] = {tpr:.4f}")

pub_ensemble = sum(norm01(pub_llr[k]) for k in SIG_KEYS) / len(SIG_KEYS)
ens_tpr = tpr_at_fpr(pub_ensemble, pub_membership)
#print(f"    Pub TPR@5%FPR [ensemble   ] = {ens_tpr:.4f}")

# Best single vs ensemble
best_tpr = max(tpr_at_fpr(pub_llr[k], pub_membership) for k in SIG_KEYS)
use_ensemble = ens_tpr >= best_tpr

# ── Priv scoring ──
# For priv, no sample was ever in shadow training → all shadow runs are OUT.
# We need an IN reference. Strategy: for each priv sample, find pub samples
# of the same class and use their shadow IN distribution as the IN reference.
# This is the per-class IN reference approach.
#print("\n    Scoring priv with per-class IN reference...")

priv_llr = {k: np.zeros(N_PRIV, dtype=np.float32) for k in SIG_KEYS}

for cls in range(NUM_CLASSES):
    pub_cls_mask  = (pub_labels_np == cls)          # pub samples of this class
    priv_cls_mask = (priv_labels_np == cls)          # priv samples of this class

    if priv_cls_mask.sum() == 0:
        continue

    for k in SIG_KEYS:
        # Build IN distribution: pub samples of same class that were IN shadow training
        # Shape: (n_pub_cls, N_SHADOW), masked to IN only
        pub_cls_shadow = shadow_pub_sigs[k][pub_cls_mask]       # (n_cls, S)
        pub_cls_in_mask = shadow_in_mask[pub_cls_mask]           # (n_cls, S)

        in_vals_cls = pub_cls_shadow[pub_cls_in_mask]            # all IN observations for this class
        global_mu_in_cls  = in_vals_cls.mean() if len(in_vals_cls) > 0 else 0.0
        global_sig_in_cls = in_vals_cls.std()  + 1e-8

        # For each priv sample of this class: all shadow runs are OUT
        priv_cls_shadow = shadow_priv_sigs[k][priv_cls_mask]    # (n_priv_cls, S)
        mu_out  = priv_cls_shadow.mean(axis=1)                   # (n_priv_cls,)
        sig_out = priv_cls_shadow.std(axis=1) + 1e-8

        target_vals = target_priv[k][priv_cls_mask]

        llr = (scipy_norm.logpdf(target_vals, global_mu_in_cls, global_sig_in_cls) -
               scipy_norm.logpdf(target_vals, mu_out, sig_out))

        priv_llr[k][priv_cls_mask] = llr.astype(np.float32)

# Final priv score
if use_ensemble:
    priv_final = sum(norm01(priv_llr[k]) for k in SIG_KEYS) / len(SIG_KEYS)
    #print(f"    Using ensemble (TPR={ens_tpr:.4f})")
else:
    best_k = max(SIG_KEYS, key=lambda k: tpr_at_fpr(pub_llr[k], pub_membership))
    priv_final = norm01(priv_llr[best_k])
    #print(f"    Using best signal: {best_k} (TPR={best_tpr:.4f})")


# ──────────────────────────────────────────────
# Step 4: Save + Submit
# ──────────────────────────────────────────────
df = pd.DataFrame({"id": priv_ids_list, "score": priv_final.tolist()})
df.to_csv(OUTPUT_CSV, index=False)

final_tpr = ens_tpr if use_ensemble else best_tpr
print(f"\n[4/4] Saved {len(df)} rows → {OUTPUT_CSV}")
print(f"      Score range    : [{df['score'].min():.4f}, {df['score'].max():.4f}]")

try:
    with open(OUTPUT_CSV, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": API_KEY},
            files={"file": (OUTPUT_CSV.name, f, "application/csv")},
            timeout=(10, 600),
        )
    body = resp.json() if "application/json" in resp.headers.get("content-type", "") \
           else {"raw": resp.text}
    resp.raise_for_status()
    print("\nSuccessfully submitted. Server:", body)
except requests.exceptions.RequestException as e:
    print(f"Submission error: {e}")