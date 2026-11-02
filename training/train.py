"""livekit-wakeword training with your own labelled clips mixed into every batch, resumable.

Subclasses the stock WakeWordTrainer (v0.2.3). The 3-phase schedule, focal loss, mixup,
label smoothing, negative-weight ramp, validation cadence and checkpoint averaging are the
stock ones; _train_phase below is a line-for-line copy of upstream's loop with only two
additions: (1) start at a saved step, (2) save {model, optimizer, step, checkpoints} every
--save-every steps. The LR / neg-weight schedules are pure functions of (step, steps), so a
resumed run follows the same curve. (Batch order after resume differs — the data generator
is a cyclic mmap reader — which is immaterial at these sizes.)

Extra batch classes: real_positive (label 1) and real_negative (label 0) from
real_pos_train.npy / real_neg_train.npy; --real-pos 0 --real-neg 0 gives the stock control.

Usage:
  python train.py <config.yaml> --tag <name> [--real-pos 25] [--real-neg 50] [--size medium] [--steps N]
Re-run the same command after a kill: it resumes. Outputs <tag>.pt and <tag>.onnx.
"""
import argparse
import copy
from pathlib import Path

import torch
from tqdm import tqdm

from livekit.wakeword.config import load_config
from livekit.wakeword.data.dataset import create_dataloader
from livekit.wakeword.export.onnx import export_onnx
from livekit.wakeword.training import trainer as T
from livekit.wakeword.training.trainer import WakeWordTrainer


class MixTrainer(WakeWordTrainer):
    def __init__(self, config, real_dir: Path, n_pos: int, n_neg: int, ckpt: Path, save_every: int):
        self._real_dir, self._n_pos, self._n_neg = real_dir, n_pos, n_neg
        self._ckpt, self._save_every = ckpt, save_every
        super().__init__(config)

    # ---------- data
    def _build_dataloader(self):
        md = self.config.model_output_dir
        files = {
            "positive": md / "positive_features_train.npy",
            "adversarial_negative": md / "negative_features_train.npy",
            "ACAV100M_sample": self.config.data_path / "features" / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
            "background_noise": md / "background_noise_features_train.npy",
        }
        n = dict(self.config.batch_n_per_class)
        if self._n_pos:
            files["real_positive"] = self._real_dir / "real_pos_train.npy"; n["real_positive"] = self._n_pos
        if self._n_neg:
            files["real_negative"] = self._real_dir / "real_neg_train.npy"; n["real_negative"] = self._n_neg
        labels = {k: (lambda _: 1) if "positive" in k else (lambda _: 0) for k in files}
        return create_dataloader(data_files={k: v for k, v in files.items() if Path(v).exists()},
                                 n_per_class=n, label_funcs=labels)

    # ---------- validation features cached once (stock re-reads ~0.5 GB every call)
    def _load_validation_data(self):
        if not hasattr(self, "_val_cache"):
            self._val_cache = super()._load_validation_data()
        return self._val_cache

    # ---------- resumable copy of upstream _train_phase
    def _train_phase(self, phase, steps, base_lr, max_negative_weight, dataloader):
        done = self._ckpt / f"phase{phase}.pt"
        mid = self._ckpt / f"phase{phase}_mid.pt"
        if done.exists():
            st = torch.load(done, map_location=self.device, weights_only=False)
            self.model.load_state_dict(st["model"]); self.checkpoints = st["checkpoints"]; self._metrics_log = st["metrics_log"]
            print(f"[resume] phase {phase} already complete", flush=True)
            return

        T.logger.info(f"=== Phase {phase}: {steps} steps, LR={base_lr}, max_neg_w={max_negative_weight} ===")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=base_lr, weight_decay=self.config.weight_decay)
        start = 0
        if mid.exists():
            st = torch.load(mid, map_location=self.device, weights_only=False)
            self.model.load_state_dict(st["model"]); optimizer.load_state_dict(st["optimizer"])
            self.checkpoints = st["checkpoints"]; self._metrics_log = st["metrics_log"]; start = st["step"]
            print(f"[resume] phase {phase} from step {start}/{steps}", flush=True)

        label_smoothing = self.config.label_smoothing
        warmup_steps = steps // 5 if phase == 1 else 0
        hold_steps = steps // 3 if phase == 1 else 0
        self.model.train()
        data_iter = iter(dataloader)
        validation_interval = max(1, steps // 20)

        pbar = tqdm(range(start, steps), desc=f"Phase {phase}", unit="step", initial=start, total=steps)
        for step in pbar:
            try:
                features, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                features, labels = next(data_iter)
            features, labels = features.to(self.device), labels.to(self.device)

            lam = torch.distributions.Beta(0.2, 0.2).sample().to(self.device)
            perm = torch.randperm(features.size(0), device=self.device)
            features = lam * features + (1 - lam) * features[perm]
            labels = lam * labels + (1 - lam) * labels[perm]
            if label_smoothing > 0:
                labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing

            predictions = self.model(features).squeeze(-1)
            loss_per_sample = T.focal_loss(predictions, labels, gamma=2.0)
            neg_weight = T._negative_weight_schedule(step, steps, max_negative_weight)
            weights = torch.where(labels < 0.5, neg_weight, 1.0)
            loss = (loss_per_sample * weights).mean()

            lr = T._cosine_warmup_schedule(step, steps, warmup_steps, hold_steps, base_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.1e}", neg_w=f"{neg_weight:.0f}")

            if step >= steps * 3 // 4 and step % validation_interval == 0:
                metrics = self._validate()
                pbar.write(f"  Validation @ step {step}: FPPH={metrics['fpph']:.2f}, "
                           f"Recall={metrics['recall']:.3f}, Acc={metrics['accuracy']:.3f}")
                self._log_metrics(step, phase, metrics)
                self._save_checkpoint(step, phase, metrics)
                self.model.train()

            if (step + 1) % self._save_every == 0 and step + 1 < steps:
                torch.save({"model": self.model.state_dict(), "optimizer": optimizer.state_dict(),
                            "checkpoints": self.checkpoints, "metrics_log": self._metrics_log,
                            "step": step + 1}, mid.with_suffix(".tmp"))
                mid.with_suffix(".tmp").replace(mid)

        torch.save({"model": self.model.state_dict(), "checkpoints": self.checkpoints,
                    "metrics_log": self._metrics_log}, done)
        mid.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--real-pos", type=int, default=0)
    ap.add_argument("--real-neg", type=int, default=0)
    ap.add_argument("--size", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=1000)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.size:
        cfg.model.model_size = a.size
    if a.steps:
        cfg.steps = a.steps
    md = cfg.model_output_dir
    if (md / f"{a.tag}.onnx").exists():
        print("already exported", md / f"{a.tag}.onnx"); return
    cfg_tag = copy.deepcopy(cfg)
    ck = md / f".ckpt_{a.tag}"
    ck.mkdir(exist_ok=True)
    t = MixTrainer(cfg, md, a.real_pos, a.real_neg, ckpt=ck, save_every=a.save_every)
    t._metrics_path = md / f"{a.tag}_metrics.json"
    t.train()
    pt = md / f"{a.tag}.pt"
    t.save(pt)
    export_onnx(cfg_tag, pt, md / f"{a.tag}.onnx")
    print("exported", md / f"{a.tag}.onnx", flush=True)


if __name__ == "__main__":
    main()
