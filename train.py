"""CSE407 training template.

Reads a YAML config, trains, and writes one folder per run:

runs/<experiment>/seed_<seed>/
    config.yaml        the exact settings used
    meta.json          status, timings, versions, best result
    history.json       loss per epoch
    train.log          printed output
    checkpoints/       last.pt (to resume), best.pt (lowest validation loss)
    plots/curves.png   training and validation loss

Run:      python train.py --config configs/demo.yaml
Seed:     python train.py --config configs/demo.yaml --seed 3
Override: python train.py --config configs/demo.yaml --set lr=0.001 --set patience=5
Resume:   run the same command again.
"""
import argparse, json, os, platform, socket, subprocess, time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
import matplotlib
matplotlib.use("Agg")  # no display on the Spark; plots go to files
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

# Every setting must appear here. A key in a YAML file that is not here is an error.
DEFAULTS = {
    "experiment": "demo",   # folder name under runs/
    "seed": 0,              # model initialisation and batch order
    "data_seed": 0,         # data generation. Keep fixed when you compare seeds.
    "epochs": 200,
    "batch_size": 128,
    "lr": 1e-2,
    "hidden": 64,
    "patience": 10,         # epochs without improvement before early stop
    "min_delta": 0.0,       # smallest drop in validation loss that counts as improvement
    "ckpt_every": 1,        # epochs between last.pt saves
    "plot_every": 5,        # epochs between curves.png saves
}


def load_config(path, seed, overrides):
    cfg = dict(DEFAULTS)
    user = yaml.safe_load(open(path)) or {}
    for item in overrides:
        key, value = item.split("=", 1)
        user[key] = yaml.safe_load(value)
    if seed is not None:
        user["seed"] = seed
    unknown = set(user) - set(DEFAULTS)
    if unknown:
        raise SystemExit(f"Unknown config keys: {sorted(unknown)}")
    cfg.update(user)
    return cfg


class EarlyStopper:
    """Stop when validation loss has not improved by min_delta for `patience` epochs."""
    def __init__(self, patience, min_delta):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.bad = float("inf"), 0

    def step(self, val_loss):
        """Return True if val_loss is a new best."""
        if val_loss < self.best - self.min_delta:
            self.best, self.bad = val_loss, 0
            return True
        self.bad += 1
        return False

    @property
    def should_stop(self):
        return self.bad >= self.patience

    def state_dict(self):
        return {"best": self.best, "bad": self.bad}

    def load_state_dict(self, s):
        self.best, self.bad = s["best"], s["bad"]


def write_atomic(path, write_fn):
    """Write to a temporary file, then rename. A crash cannot leave a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    write_fn(tmp)
    os.replace(tmp, path)


def save_json(obj, path):
    write_atomic(path, lambda p: p.write_text(json.dumps(obj, indent=2)))


def save_ckpt(obj, path):
    write_atomic(path, lambda p: torch.save(obj, p))


def plot_curves(history, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["epoch"], history["train_loss"], label="train")
    ax.plot(history["epoch"], history["val_loss"], label="validation")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    a = p.parse_args()
    cfg = load_config(a.config, a.seed, a.set)

    run = ROOT / "runs" / cfg["experiment"] / f"seed_{cfg['seed']}"
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run / "plots").mkdir(exist_ok=True)

    def log(msg):
        print(msg, flush=True)
        with open(run / "train.log", "a") as f:
            f.write(msg + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # RUN RECORD: save the exact settings of this run to config.yaml
    (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    # RUN RECORD: meta.json holds status, versions, host, git commit and best result.
    # It is rewritten after every epoch so a crashed run still shows how far it got.
    meta = {
        "experiment": cfg["experiment"], "seed": cfg["seed"], "status": "running",
        "started": time.strftime("%Y-%m-%d %H:%M:%S"), "finished": None,
        "host": socket.gethostname(), "python": platform.python_version(),
        "torch": torch.__version__, "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "git_commit": git_commit(), "config": cfg,
        "epochs_run": 0, "best_epoch": None, "best_val_loss": None,
    }
    save_json(meta, run / "meta.json")

    # ---- Replace this block with your data and model ----
    g = torch.Generator().manual_seed(cfg["data_seed"])
    X = torch.randn(2000, 20, generator=g); w = torch.randn(20, 1, generator=g)
    y = X @ w + 0.5 * torch.randn(2000, 1, generator=g)
    Xtr, ytr, Xva, yva = (t.to(device) for t in (X[:1600], y[:1600], X[1600:], y[1600:]))
    torch.manual_seed(cfg["seed"])
    model = nn.Sequential(nn.Linear(20, cfg["hidden"]), nn.ReLU(), nn.Linear(cfg["hidden"], 1)).to(device)
    # -----------------------------------------------------

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()
    # EARLY STOPPING: create the stopper. It counts epochs without improvement.
    stopper = EarlyStopper(cfg["patience"], cfg["min_delta"])
    # CURVES: loss values are stored here each epoch; this dict is what we plot and save
    history = {"epoch": [], "train_loss": [], "val_loss": []}
    start = 0

    # CHECKPOINT (LOAD): if last.pt exists, restore the model, optimizer, stopper
    # and history, and continue from the next epoch. This is how a run resumes.
    last = run / "checkpoints" / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        stopper.load_state_dict(ck["stopper"]); history = ck["history"]
        start = ck["epoch"] + 1
        log(f"Resumed at epoch {start}")

    stopped_early = False
    for epoch in range(start, cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(Xtr), cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            opt.zero_grad(); loss_fn(model(Xtr[idx]), ytr[idx]).backward(); opt.step()

        # VALIDATION: measure loss on training and validation data (no gradient updates)
        model.eval()
        with torch.no_grad():
            tr, va = loss_fn(model(Xtr), ytr).item(), loss_fn(model(Xva), yva).item()
        # CURVES: record this epoch's losses
        history["epoch"].append(epoch); history["train_loss"].append(tr); history["val_loss"].append(va)
        log(f"epoch {epoch:3d}  train {tr:.4f}  val {va:.4f}")

        # EARLY STOPPING + BEST CHECKPOINT: stopper.step returns True when the validation
        # loss is a new best. Then save best.pt (weights only) and note the result in meta.
        if stopper.step(va):
            save_ckpt({"model": model.state_dict(), "epoch": epoch, "val_loss": va},
                      run / "checkpoints" / "best.pt")
            meta["best_epoch"], meta["best_val_loss"] = epoch, va
        # CHECKPOINT (SAVE): every ckpt_every epochs save everything needed to resume
        if epoch % cfg["ckpt_every"] == 0:
            save_ckpt({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                       "stopper": stopper.state_dict(), "history": history}, last)
        # CURVES (DATA): write the loss numbers to history.json every epoch
        save_json(history, run / "history.json")
        # CURVES (PLOT): redraw plots/curves.png every plot_every epochs
        if epoch % cfg["plot_every"] == 0:
            plot_curves(history, run / "plots" / "curves.png")
        # RUN RECORD: update meta.json
        meta["epochs_run"] = epoch + 1
        save_json(meta, run / "meta.json")

        # EARLY STOPPING: leave the loop when there has been no improvement for `patience` epochs
        if stopper.should_stop:
            stopped_early = True
            log(f"Early stop at epoch {epoch}. Best validation loss {stopper.best:.4f}")
            break

    # CURVES (PLOT): final plot, so the last epochs always appear
    plot_curves(history, run / "plots" / "curves.png")
    # RUN RECORD: mark the run as finished or early_stopped
    meta["status"] = "early_stopped" if stopped_early else "finished"
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(meta, run / "meta.json")
    log(f"Run folder: {run}")


if __name__ == "__main__":
    main()
