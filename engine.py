from __future__ import annotations

import json
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, get_worker_info
from tqdm import tqdm

from losses import soft_dice_score
from metrics import dice_from_labels, logits_to_labels, save_summary, summarize_case_metrics
from preprocess import crop_or_pad


class PolyLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = -1):
        self.initial_lr = float(initial_lr)
        self.max_steps = int(max_steps)
        self.exponent = float(exponent)
        super().__init__(optimizer, last_epoch=current_step)

    def get_lr(self):
        step = max(self.last_epoch, 0)
        factor = (1.0 - step / max(self.max_steps, 1)) ** self.exponent
        return [self.initial_lr * factor for _ in self.optimizer.param_groups]


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    worker_info = get_worker_info()
    if worker_info is not None and hasattr(worker_info.dataset, "set_worker_seed"):
        worker_info.dataset.set_worker_seed(worker_seed)


def build_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    shuffle: bool,
    seed: int | None = None,
) -> DataLoader:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=shuffle,
        worker_init_fn=_seed_worker if seed is not None else None,
        generator=generator,
    )


def _move_batch(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    return images, labels


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, grad_clip: float, amp_enabled: bool) -> dict:
    model.train()
    losses = []
    for batch in tqdm(loader, desc="train", leave=False):
        images, labels = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled and device.type == "cuda"):
            logits = model(images)
            loss = loss_fn(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {"loss": float(np.mean(losses)) if losses else 0.0}


@torch.no_grad()
def validate_patches(model, loader, loss_fn, device, num_classes: int, amp_enabled: bool) -> dict:
    model.eval()
    losses = []
    dices = []
    for batch in tqdm(loader, desc="val-patch", leave=False):
        images, labels = _move_batch(batch, device)
        with torch.autocast(device_type=device.type, enabled=amp_enabled and device.type == "cuda"):
            logits = model(images)
            loss = loss_fn(logits, labels)
        losses.append(float(loss.detach().cpu()))
        dice = soft_dice_score(logits, labels, num_classes=num_classes).mean()
        dices.append(float(dice.detach().cpu()))
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "pseudo_dice": float(np.mean(dices)) if dices else 0.0,
    }


def _window_starts(length: int, patch: int, overlap: float) -> list[int]:
    if length <= patch:
        return [0]
    if overlap <= 0.0:
        starts = list(range(0, length - patch + 1, patch))
        if starts[-1] != length - patch:
            starts.append(length - patch)
        return starts
    stride = max(1, int(round(patch * (1.0 - overlap))))
    starts = list(range(0, max(length - patch + 1, 1), stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


@torch.no_grad()
def predict_volume(model, image: torch.Tensor, patch_size: list[int], overlap: float, device: torch.device) -> np.ndarray:
    model.eval()
    image = image.squeeze(0)
    spatial_shape = image.shape[1:]
    patch_size = list(patch_size)
    starts = [_window_starts(spatial_shape[i], patch_size[i], overlap) for i in range(3)]

    if overlap <= 0.0:
        prediction = np.zeros(spatial_shape, dtype=np.uint8)
        for start_d in starts[0]:
            for start_h in starts[1]:
                for start_w in starts[2]:
                    bbox = [
                        (start_d, start_d + patch_size[0]),
                        (start_h, start_h + patch_size[1]),
                        (start_w, start_w + patch_size[2]),
                    ]
                    patch = crop_or_pad(image.cpu().numpy(), bbox, fill_value=0.0)
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).float().to(device)
                    logits = model(patch_tensor)
                    labels = logits_to_labels(logits).squeeze(0).cpu().numpy().astype(np.uint8)
                    end_d = min(start_d + patch_size[0], spatial_shape[0])
                    end_h = min(start_h + patch_size[1], spatial_shape[1])
                    end_w = min(start_w + patch_size[2], spatial_shape[2])
                    prediction[start_d:end_d, start_h:end_h, start_w:end_w] = labels[: end_d - start_d, : end_h - start_h, : end_w - start_w]
        return prediction

    num_classes = int(model.seg_head.conv.out_channels)
    logits_sum = torch.zeros((num_classes, *spatial_shape), dtype=torch.float16)
    count_map = torch.zeros(spatial_shape, dtype=torch.float16)

    for start_d in starts[0]:
        for start_h in starts[1]:
            for start_w in starts[2]:
                bbox = [
                    (start_d, start_d + patch_size[0]),
                    (start_h, start_h + patch_size[1]),
                    (start_w, start_w + patch_size[2]),
                ]
                patch = crop_or_pad(image.cpu().numpy(), bbox, fill_value=0.0)
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).float().to(device)
                logits = model(patch_tensor).squeeze(0).cpu().to(torch.float16)
                end_d = min(start_d + patch_size[0], spatial_shape[0])
                end_h = min(start_h + patch_size[1], spatial_shape[1])
                end_w = min(start_w + patch_size[2], spatial_shape[2])
                logits_sum[:, start_d:end_d, start_h:end_h, start_w:end_w] += logits[:, : end_d - start_d, : end_h - start_h, : end_w - start_w]
                count_map[start_d:end_d, start_h:end_h, start_w:end_w] += 1

    logits_avg = logits_sum / torch.clamp(count_map.unsqueeze(0), min=1)
    return torch.argmax(logits_avg, dim=0).cpu().numpy().astype(np.uint8)


@torch.no_grad()
def validate_full_volumes(model, loader, device, num_classes: int, patch_size: list[int], overlap: float, output_dir: str | Path) -> dict:
    case_metrics = []
    inference_times_ms = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc="val-full", leave=False):
        image = batch["image"][0]
        label = batch["label"][0, 0].numpy()
        case_id = batch["case_id"][0]
        start_time = perf_counter()
        prediction = predict_volume(model, image.unsqueeze(0), patch_size, overlap, device)
        inference_times_ms.append((perf_counter() - start_time) * 1000.0)
        dice_scores = dice_from_labels(prediction, label, num_classes=num_classes)
        case_metrics.append(
            {
                "case_id": case_id,
                "dice_per_class": dice_scores,
                "mean_dice": float(np.mean(dice_scores)),
            }
        )

    summary = summarize_case_metrics(case_metrics)
    summary["infer_time_ms_per_case"] = float(np.mean(inference_times_ms)) if inference_times_ms else 0.0
    save_summary(summary, output_dir / "summary.json")
    return summary


def append_metrics(log_file: str | Path, payload: dict) -> None:
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
