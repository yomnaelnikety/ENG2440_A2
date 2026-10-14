import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, get_worker_info

DEFAULT_CFG = dict(
    sampler="uniform",
    p_tumour=0.5,
    patch_size=(96, 96, 96),
    batch_size=2,
    iters=5000,
    lr=1e-3,
    weight_decay=1e-5,
    val_every=500,
    log_every=25,
    num_workers=8,
    seed=42,
    deterministic=True,
    amp=True,
    base_filters=16,
    depth=4,
    cache_root="cache_preproc",
    out_dir="runs/uniform",
    max_val_cases=None,
    val_overlap=0.0,
    val_batch_size=4,

LOSS_DESCRIPTION = "BCE-with-logits + (1 - soft Dice)"
OPTIMIZER_DESCRIPTION = "AdamW, cosine decay to 1% of the peak learning rate"

# Seeding
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Cache access
def load_cache_index(cache_root):
    return pd.read_csv(Path(cache_root) / "cache_index.csv")


def open_case(cache_root, case_id):
    img = np.load(Path(cache_root) / f"{case_id}_img.npy", mmap_mode="r")
    lab = np.load(Path(cache_root) / f"{case_id}_lab.npy", mmap_mode="r")
    return img, lab

# Augmentation 
class Augment3D:
    def __init__(self, p_flip=0.5, scale=(0.9, 1.1), shift=0.1, noise_std=0.05):
        self.p_flip = p_flip
        self.scale = scale
        self.shift = shift
        self.noise_std = noise_std

    def __call__(self, img, msk, rng):
        if rng.random() < self.p_flip:
            img, msk = img[:, ::-1], msk[:, ::-1]   
        C = img.shape[0]
        brain = img != 0
        s = rng.uniform(*self.scale, size=(C, 1, 1, 1)).astype(np.float32)
        b = rng.uniform(-self.shift, self.shift, size=(C, 1, 1, 1)).astype(np.float32)
        img = np.where(brain, img * s + b, 0).astype(np.float32)
        noise = rng.normal(0, self.noise_std, img.shape).astype(np.float32)
        img = img + np.where(brain, noise, 0)
        return np.ascontiguousarray(img), np.ascontiguousarray(msk)

# Patch reading
def read_patch(img_mm, lab_mm, start, size):
    sl = tuple(slice(int(s), int(s) + int(n)) for s, n in zip(start, size))
    img = np.asarray(img_mm[(slice(None),) + sl], dtype=np.float32)
    lab = (np.asarray(lab_mm[sl]) > 0).astype(np.float32)[None]
    pad = [(0, 0)] + [(0, int(n) - got) for n, got in zip(size, img.shape[1:])]
    if any(p[1] > 0 for p in pad):
        img = np.pad(img, pad)
        lab = np.pad(lab, pad)
    return img, lab

class PatchDataset(Dataset):
    def __init__(self, cache_root, case_ids, patch_size, n_samples, sampler="uniform", p_tumour=0.5,
                 augment=True, seed=0, max_tumour_coords=4000):
        assert sampler in ("uniform", "tumour")
        self.cache_root = Path(cache_root)
        self.case_ids = list(case_ids)
        self.patch_size = tuple(int(p) for p in patch_size)
        self.n_samples = int(n_samples)
        self.sampler = sampler
        self.p_tumour = p_tumour
        self.augment = Augment3D() if augment else None
        self.rng = np.random.default_rng(seed)
        self._open = {}

        self.shapes = {}
        self.tumour_coords = {}
        coord_rng = np.random.default_rng(seed)
        for cid in self.case_ids:
            _, lab = open_case(self.cache_root, cid)
            self.shapes[cid] = tuple(lab.shape)
            if sampler == "tumour":
                coords = np.argwhere(lab > 0)
                assert len(coords) > 0, f"{cid} has no tumour voxels"
                if len(coords) > max_tumour_coords:
                    coords = coords[coord_rng.choice(len(coords), max_tumour_coords, replace=False)]
                self.tumour_coords[cid] = coords.astype(np.int32)

    def __len__(self):
        return self.n_samples

    def _get(self, cid):
        if cid not in self._open:
            self._open[cid] = open_case(self.cache_root, cid)
        return self._open[cid]

    def __getitem__(self, idx):
        rng = self.rng
        cid = self.case_ids[int(rng.integers(len(self.case_ids)))]
        img_mm, lab_mm = self._get(cid)
        size = np.array(self.patch_size)
        shape = np.array(self.shapes[cid])
        max_start = np.maximum(shape - size, 0)

        use_tumour = (self.sampler == "tumour") and (rng.random() < self.p_tumour)
        if use_tumour:
            coords = self.tumour_coords[cid]
            centre = coords[int(rng.integers(len(coords)))].astype(np.int64)
            quarter = size // 4
            jitter = rng.integers(-quarter, quarter + 1)
            start = centre + jitter - size // 2
        else:
            start = rng.integers(0, max_start + 1)
        start = np.clip(start, 0, max_start)

        img, lab = read_patch(img_mm, lab_mm, start, size)
        if self.augment is not None:
            img, lab = self.augment(img, lab, rng)
        img = np.ascontiguousarray(img.transpose(0, 3, 1, 2))
        lab = np.ascontiguousarray(lab.transpose(0, 3, 1, 2))
        return torch.from_numpy(img), torch.from_numpy(lab)


def seed_worker(worker_id):
    info = get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed)


# Model: lightweight 3D U-Net (fully convolutional, so it can run on any input whose size is divisible by 2^(depth-1)).
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False),
            nn.InstanceNorm3d(cout, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False),
            nn.InstanceNorm3d(cout, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, base=16, depth=4):
        super().__init__()
        chans = [base * 2 ** i for i in range(depth)]
        self.encoders = nn.ModuleList(
            [ConvBlock(in_channels if i == 0 else chans[i - 1], chans[i]) for i in range(depth)])
        self.downs = nn.ModuleList([nn.Conv3d(chans[i], chans[i], 2, stride=2) for i in range(depth - 1)])
        self.ups = nn.ModuleList(
            [nn.ConvTranspose3d(chans[i + 1], chans[i], 2, stride=2) for i in reversed(range(depth - 1))])
        self.decoders = nn.ModuleList(
            [ConvBlock(2 * chans[i], chans[i]) for i in reversed(range(depth - 1))])
        self.head = nn.Conv3d(chans[0], out_channels, 1)              

    def forward(self, x):
        skips = []
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            if i < len(self.downs):
                skips.append(x)                                        
                x = self.downs[i](x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)
        return self.head(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def weights_hash(model):
    h = hashlib.md5()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()


# Loss
def soft_dice(prob, target, smooth=1.0):
    inter = (prob * target).sum()
    return (2.0 * inter + smooth) / (prob.sum() + target.sum() + smooth)


def seg_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice_loss = 1.0 - soft_dice(torch.sigmoid(logits), target)
    return bce + dice_loss, bce, dice_loss


def seg_loss_from_prob(prob, target):
    p = prob.clamp(1e-6, 1 - 1e-6)
    bce = F.binary_cross_entropy(p, target)
    return bce + (1.0 - soft_dice(prob, target))

# Sliding-window inference
# 
def tile_starts(n, size, stride):
    if n <= size:
        return [0]
    starts = list(range(0, n - size + 1, stride))
    if starts[-1] != n - size:
        starts.append(n - size)
    return starts


@torch.no_grad()
def sliding_window_prob(model, img, patch_size, overlap, device, batch_size=4, use_amp=True):
    model.eval()
    x = np.asarray(img, dtype=np.float32)
    orig = x.shape[1:]
    pad = [(0, 0)] + [(0, max(0, int(p) - s)) for p, s in zip(patch_size, orig)]
    x = np.pad(x, pad)                                              
    x = np.ascontiguousarray(x.transpose(0, 3, 1, 2))                
    patch = (int(patch_size[2]), int(patch_size[0]), int(patch_size[1])) 
    shape = x.shape[1:]
    stride = [max(1, int(round(p * (1.0 - overlap)))) for p in patch]
    starts = [tile_starts(n, p, s) for n, p, s in zip(shape, patch, stride)]
    corners = [(a, b, c) for a in starts[0] for b in starts[1] for c in starts[2]]

    xt = torch.from_numpy(x)
    prob_sum = torch.zeros(shape, dtype=torch.float32, device=device)
    count = torch.zeros(shape, dtype=torch.float32, device=device)
    for i in range(0, len(corners), batch_size):
        chunk = corners[i:i + batch_size]
        batch = torch.stack([xt[:, a:a + patch[0], b:b + patch[1], c:c + patch[2]] for a, b, c in chunk])
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp and device.type == "cuda"):
            logits = model(batch)
        probs = torch.sigmoid(logits.float())[:, 0]                   
        for (a, b, c), p in zip(chunk, probs):
            prob_sum[a:a + patch[0], b:b + patch[1], c:c + patch[2]] += p
            count[a:a + patch[0], b:b + patch[1], c:c + patch[2]] += 1.0
    prob = prob_sum / count                                              
    prob = prob.permute(1, 2, 0)                                        
    return prob[:orig[0], :orig[1], :orig[2]]                         


# Evaluation on whole volumes
def confusion_metrics(pred, target, voxel_ml=1e-3):
    tp = int((pred & target).sum().item())
    fp = int((pred & ~target).sum().item())
    fn = int((~pred & target).sum().item())
    return dict(
        dice=2.0 * tp / (2.0 * tp + fp + fn + 1e-8),
        sensitivity=tp / (tp + fn + 1e-8),
        fp_ml=fp * voxel_ml,          
        tp=tp, fp=fp, fn=fn,
    )


@torch.no_grad()
def evaluate(model, cache_root, case_ids, patch_size, device, overlap=0.0, batch_size=4, thr=0.5,
             voxel_ml=1e-3, use_amp=True):
    rows = []
    for cid in case_ids:
        img_mm, lab_mm = open_case(cache_root, cid)
        prob = sliding_window_prob(model, img_mm, patch_size, overlap, device, batch_size, use_amp)
        target = torch.from_numpy((np.asarray(lab_mm) > 0).astype(np.float32)).to(device)
        loss = float(seg_loss_from_prob(prob, target).item())
        m = confusion_metrics(prob > thr, target > 0.5, voxel_ml)
        m.update(case_id=cid, loss=loss)
        rows.append(m)
    return pd.DataFrame(rows)

# Training
def train(cfg_in):
    cfg = dict(DEFAULT_CFG)
    cfg.update(cfg_in)
    cfg["patch_size"] = tuple(int(p) for p in cfg["patch_size"])
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    cache_root = Path(cfg["cache_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["amp"]) and device.type == "cuda"

    set_seed(cfg["seed"])
    torch.backends.cudnn.deterministic = bool(cfg["deterministic"])
    torch.backends.cudnn.benchmark = not bool(cfg["deterministic"])

    index = load_cache_index(cache_root)
    train_ids = index[index.split == "train"].case_id.tolist()
    val_ids = index[index.split == "validation"].case_id.tolist()
    if cfg["max_val_cases"]:
        val_ids = val_ids[: int(cfg["max_val_cases"])]

    model = UNet3D(4, 1, cfg["base_filters"], cfg["depth"]).to(device)
    init_hash = weights_hash(model)
    n_params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["iters"], eta_min=cfg["lr"] * 0.01)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    dataset = PatchDataset(cache_root, train_ids, cfg["patch_size"], n_samples=cfg["iters"] * cfg["batch_size"],
                           sampler=cfg["sampler"], p_tumour=cfg["p_tumour"], augment=True, seed=cfg["seed"])
    gen = torch.Generator()
    gen.manual_seed(cfg["seed"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"],
                        worker_init_fn=seed_worker, generator=gen, pin_memory=(device.type == "cuda"),
                        drop_last=True, prefetch_factor=2 if cfg["num_workers"] > 0 else None)

    print(f"[{cfg['sampler']}] device={device} params={n_params:,} init_hash={init_hash[:12]} "
          f"train_cases={len(train_ids)} val_cases={len(val_ids)}", flush=True)

    history, train_log = [], []
    interval_loss, interval_n = 0.0, 0
    log_loss, log_n = 0.0, 0
    tumour_patches, total_patches = 0, 0
    best_dice, best_iter = -1.0, 0
    t0 = time.time()
    model.train()

    for it, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        loss, bce, dice_loss = seg_loss(logits.float(), y) 

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        interval_loss += float(loss.item()); interval_n += 1
        log_loss += float(loss.item()); log_n += 1
        tumour_patches += int((y.flatten(1).sum(1) > 0).sum().item())
        total_patches += int(y.shape[0])

        if it % cfg["log_every"] == 0:
            train_log.append(dict(iter=it, train_loss=log_loss / log_n, lr=scheduler.get_last_lr()[0],
                                  tumour_patch_frac=tumour_patches / total_patches))
            log_loss, log_n = 0.0, 0

        if it % cfg["val_every"] == 0 or it == cfg["iters"]:
            val = evaluate(model, cache_root, val_ids, cfg["patch_size"], device, overlap=cfg["val_overlap"],
                           batch_size=cfg["val_batch_size"], use_amp=use_amp)
            model.train()                                                
            row = dict(iter=it, train_loss=interval_loss / max(interval_n, 1), val_loss=val.loss.mean(),
                       val_dice=val.dice.mean(), val_dice_median=val.dice.median(),
                       val_sens=val.sensitivity.mean(), val_fp_ml=val.fp_ml.mean(),
                       lr=scheduler.get_last_lr()[0], tumour_patch_frac=tumour_patches / total_patches,
                       minutes=(time.time() - t0) / 60.0)
            history.append(row)
            interval_loss, interval_n = 0.0, 0
            pd.DataFrame(history).to_csv(out / "history.csv", index=False)
            pd.DataFrame(train_log).to_csv(out / "train_log.csv", index=False)
            if row["val_dice"] > best_dice: 
                best_dice, best_iter = row["val_dice"], it
                torch.save(dict(model=model.state_dict(), cfg=cfg, iter=it, val_dice=best_dice), out / "best.pt")
            print(f"[{cfg['sampler']}] it {it:5d}/{cfg['iters']} | train {row['train_loss']:.4f} | "
                  f"val loss {row['val_loss']:.4f} | val Dice {row['val_dice']:.4f} | "
                  f"sens {row['val_sens']:.3f} | FP {row['val_fp_ml']:.1f} mL | "
                  f"tumour patches {100 * row['tumour_patch_frac']:.0f}% | {row['minutes']:.1f} min", flush=True)

    torch.save(dict(model=model.state_dict(), cfg=cfg, iter=cfg["iters"]), out / "last.pt")
    pd.DataFrame(train_log).to_csv(out / "train_log.csv", index=False)
    summary = dict(
        cfg={k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        sampler=cfg["sampler"], params=n_params, init_hash=init_hash, loss=LOSS_DESCRIPTION,
        optimizer=OPTIMIZER_DESCRIPTION, lr=cfg["lr"], iters=cfg["iters"], batch_size=cfg["batch_size"],
        patches_seen=cfg["iters"] * cfg["batch_size"], patch_size=list(cfg["patch_size"]), seed=cfg["seed"],
        best_iter=best_iter, best_val_dice=best_dice,
        tumour_patch_frac_train=tumour_patches / max(total_patches, 1),
        minutes=(time.time() - t0) / 60.0, device=str(device),
        gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        max_gpu_mem_gb=(torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else None,
        torch=torch.__version__, python=platform.python_version(),
    )
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{cfg['sampler']}] finished. best val Dice {best_dice:.4f} at iteration {best_iter}.", flush=True)
    return summary
