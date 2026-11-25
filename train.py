"""CSE407 training runner for smoke tests and CIFAR-10 robustness experiments.

Reads a YAML config, trains, and writes one folder per run:

runs/<experiment>/seed_<seed>/
    config.yaml        the exact settings used
    meta.json          status, timings, versions, best result
    history.json       loss per epoch
    train.log          printed output
    checkpoints/       last.pt (to resume), best.pt (lowest validation loss)
    plots/curves.png   training and validation loss

Run:      python train.py --config configs/demo.yaml
Project:  python train.py --config configs/project.yaml
Seed:     python train.py --config configs/demo.yaml --seed 3
Override: python train.py --config configs/demo.yaml --set lr=0.001 --set patience=5
Resume:   run the same command again.
"""
import argparse, csv, json, os, platform, random, socket, subprocess, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import yaml
import matplotlib
matplotlib.use("Agg")  # no display on the Spark; plots go to files
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

# Every setting must appear here. A key in a YAML file that is not here is an error.
DEFAULTS = {
    "experiment": "TestRun",   # folder name under runs/
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
    "dataset": "synthetic", # synthetic smoke-test data or cifar10
    "models": ["baseline_cnn", "baseline_vit", "augmented_cnn", "augmented_vit"],
    "data_dir": "data",
    "download": True,
    "require_cuda": False,
    "validation_fraction": 0.1,
    "num_workers": 2,
    "cnn_epochs": 15,
    "vit_epochs": 20,
    "cnn_lr": 1e-3,
    "vit_lr": 3e-4,
    "weight_decay": 0.01,
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


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class VisionTransformer(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_classes=10,
                 embed_dim=192, depth=6, num_heads=6):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.pos_embedding
        x = self.norm(self.transformer(x))
        return self.classifier(x[:, 0])


def corrupt_image(image, corruption, severity, generator=None):
    if corruption == "Gaussian_Noise":
        std = {1: 0.10, 2: 0.25}[severity]
        noise = torch.randn(
            image.shape, generator=generator, dtype=image.dtype
        ) * std
        return (image + noise).clamp(0, 1)
    if corruption == "Gaussian_Blur":
        sigma = {1: 0.8, 2: 1.8}[severity]
        return transforms.functional.gaussian_blur(image, kernel_size=5, sigma=sigma)
    if corruption == "Pixelation":
        size = {1: 16, 2: 8}[severity]
        small = F.interpolate(
            image.unsqueeze(0), size=(size, size), mode="nearest"
        )
        return F.interpolate(
            small, size=(32, 32), mode="nearest"
        ).squeeze(0)
    raise ValueError(f"Unknown corruption: {corruption}")


def filter_image(image, corruption):
    if corruption == "Gaussian_Noise":
        return transforms.functional.gaussian_blur(
            image, kernel_size=3, sigma=0.8
        )
    if corruption == "Gaussian_Blur":
        # Create a blurred reference using the average corruption sigma (~1.3)
        blurred = transforms.functional.gaussian_blur(
            image, kernel_size=5, sigma=1.3
        )
        # Unsharp Masking: image + strength * (image - blurred)
        strength = 1.5
        sharpened = image + strength * (image - blurred)
        return sharpened.clamp(0, 1)
    if corruption == "Pixelation":
        padding = 1
        padded = F.pad(
            image.unsqueeze(0), (padding, padding, padding, padding), mode="reflect"
        )
        patches = padded.unfold(2, 3, 1).unfold(3, 3, 1)
        return patches.flatten(-2).median(dim=-1).values.squeeze(0)
    return image


class CorruptedDataset(Dataset):
    def __init__(self, dataset, corruption, severity, filtered=False, seed=0):
        self.dataset = dataset
        self.corruption = corruption
        self.severity = severity
        self.filtered = filtered
        self.seed = seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        generator = torch.Generator().manual_seed(self.seed + index)
        image = corrupt_image(
            image, self.corruption, self.severity, generator=generator
        )
        if self.filtered:
            image = filter_image(image, self.corruption)
        return image, label


class CorruptedTrainingDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if random.random() < 0.5:
            return image, label
        corruption = random.choice(
            ("Gaussian_Noise", "Gaussian_Blur", "Pixelation")
        )
        severity = random.choice((1, 2))
        return corrupt_image(image, corruption, severity), label


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


def make_cifar_model(name):
    return VisionTransformer() if "vit" in name else CNN()


def cifar_training_settings(name, cfg):
    is_vit = "vit" in name
    epochs = cfg["vit_epochs"] if is_vit else cfg["cnn_epochs"]
    lr = cfg["vit_lr"] if is_vit else cfg["cnn_lr"]
    return epochs, lr


def make_cifar_loaders(cfg):
    data_dir = Path(cfg["data_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    if not 0 < cfg["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tensor_transform = transforms.ToTensor()
    train_data = datasets.CIFAR10(
        root=str(data_dir), train=True, download=cfg["download"],
        transform=train_transform,
    )
    validation_data = datasets.CIFAR10(
        root=str(data_dir), train=True, download=False, transform=tensor_transform,
    )
    test_data = datasets.CIFAR10(
        root=str(data_dir), train=False, download=cfg["download"],
        transform=tensor_transform,
    )

    validation_size = int(len(train_data) * cfg["validation_fraction"])
    if validation_size <= 0 or validation_size >= len(train_data):
        raise ValueError("validation_fraction must be between 0 and 1")
    split_generator = torch.Generator().manual_seed(cfg["data_seed"])
    indices = torch.randperm(len(train_data), generator=split_generator).tolist()
    train_indices = indices[validation_size:]
    validation_indices = indices[:validation_size]

    train_subset = Subset(train_data, train_indices)
    validation_subset = Subset(validation_data, validation_indices)
    return train_subset, validation_subset, test_data


def train_cifar_model(name, train_subset, validation_subset, cfg, device,
                      run, log, meta):
    model_dir = run / "checkpoints" / name
    model_dir.mkdir(parents=True, exist_ok=True)
    model = make_cifar_model(name).to(device)
    epochs, learning_rate = cifar_training_settings(name, cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=cfg["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()
    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "train_accuracy": [], "val_accuracy": []}
    stopper = EarlyStopper(cfg["patience"], cfg["min_delta"])
    start = 0
    last = model_dir / "last.pt"
    best = model_dir / "best.pt"

    if last.exists():
        checkpoint = torch.load(last, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        if (checkpoint.get("complete")
                and (checkpoint.get("stopped_early")
                     or checkpoint["epoch"] + 1 >= epochs)):
            best_checkpoint = torch.load(best, map_location=device, weights_only=True)
            model.load_state_dict(best_checkpoint["model"])
            return (
                model, checkpoint["history"], checkpoint["best_val_loss"],
                checkpoint["best_epoch"],
            )
        optimizer.load_state_dict(checkpoint["opt"])
        scaler.load_state_dict(checkpoint["scaler"])
        stopper.load_state_dict(checkpoint["stopper"])
        history = checkpoint["history"]
        start = checkpoint["epoch"] + 1
        log(f"Resumed {name} at epoch {start}")

    train_data = train_subset
    if name.startswith("augmented_"):
        train_data = CorruptedTrainingDataset(train_subset)
    train_generator = torch.Generator().manual_seed(cfg["seed"])
    common = {
        "batch_size": cfg["batch_size"],
        "num_workers": cfg["num_workers"],
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_data, shuffle=True, generator=train_generator, **common
    )
    validation_loader = DataLoader(validation_subset, shuffle=False, **common)
    stopped_early = False

    for epoch in range(start, epochs):
        model.train()
        train_loss = train_correct = train_total = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            count = labels.size(0)
            train_loss += loss.item() * count
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += count

        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast(
                    device_type=device.type, enabled=device.type == "cuda"
                ):
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                count = labels.size(0)
                val_loss += loss.item() * count
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += count

        train_loss /= train_total
        val_loss /= val_total
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        log(
            f"{name} epoch {epoch + 1:3d}/{epochs}  "
            f"train loss {train_loss:.4f} acc {train_accuracy:.2f}%  "
            f"val loss {val_loss:.4f} acc {val_accuracy:.2f}%"
        )

        if stopper.step(val_loss):
            save_ckpt(
                {"model": model.state_dict(), "epoch": epoch,
                 "val_loss": val_loss, "val_accuracy": val_accuracy},
                best,
            )
        if (epoch + 1) % cfg["ckpt_every"] == 0:
            save_ckpt(
                {"model": model.state_dict(), "opt": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "epoch": epoch,
                 "stopper": stopper.state_dict(), "history": history,
                 "best_val_loss": stopper.best, "best_epoch": history["epoch"][
                     history["val_loss"].index(stopper.best)
                 ] - 1},
                last,
            )
        if (epoch + 1) % cfg["plot_every"] == 0:
            plot_curves(history, run / "plots" / f"curves_{name}.png")
        model_histories = meta.setdefault("model_histories", {})
        model_histories[name] = history
        meta["epochs_run"] = sum(
            len(item["epoch"]) for item in model_histories.values()
        )
        save_json(model_histories, run / "history.json")
        save_json(meta, run / "meta.json")

        if stopper.should_stop:
            stopped_early = True
            log(f"Early stop {name} at epoch {epoch + 1}. Best val loss {stopper.best:.4f}")
            break

    best_checkpoint = torch.load(best, map_location=device, weights_only=True)
    best_epoch = best_checkpoint["epoch"]
    best_val_loss = best_checkpoint["val_loss"]
    save_ckpt(
        {"model": model.state_dict(), "opt": optimizer.state_dict(),
         "scaler": scaler.state_dict(), "epoch": history["epoch"][-1] - 1,
         "stopper": stopper.state_dict(), "history": history,
         "best_val_loss": best_val_loss, "best_epoch": best_epoch,
         "complete": True, "stopped_early": stopped_early},
        last,
    )
    plot_curves(history, run / "plots" / f"curves_{name}.png")
    model.load_state_dict(best_checkpoint["model"])
    return model, history, best_val_loss, best_epoch


def evaluate_cifar_models(models, test_data, cfg, device, run, log):
    common = {
        "batch_size": cfg["batch_size"],
        "num_workers": cfg["num_workers"],
        "pin_memory": device.type == "cuda",
    }
    conditions = [("Clean", test_data)]
    for corruption in ("Gaussian_Noise", "Gaussian_Blur", "Pixelation"):
        for severity in (1, 2):
            raw_name = f"{corruption}_S{severity}"
            conditions.append((
                f"Raw - {raw_name}",
                CorruptedDataset(
                    test_data, corruption, severity, seed=cfg["data_seed"]
                ),
            ))
            conditions.append((
                f"Strategy A - {raw_name}",
                CorruptedDataset(
                    test_data, corruption, severity, filtered=True,
                    seed=cfg["data_seed"],
                ),
            ))

    rows = []
    for condition, dataset in conditions:
        loader = DataLoader(dataset, shuffle=False, **common)
        correct = {name: 0 for name in models}
        total = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            total += labels.size(0)
            with torch.no_grad():
                for name, model in models.items():
                    model.eval()
                    with torch.amp.autocast(
                        device_type=device.type, enabled=device.type == "cuda"
                    ):
                        predictions = model(images).argmax(dim=1)
                    correct[name] += (predictions == labels).sum().item()
        for name in models:
            accuracy = 100 * correct[name] / total
            rows.append({"Model": name, "Condition": condition, "Accuracy": accuracy})
            log(f"Evaluation {name} | {condition}: {accuracy:.2f}%")
    write_project_reports(rows, run)
    return rows


def write_project_reports(rows, run):
    results_dir = run / "results"
    results_dir.mkdir(exist_ok=True)
    models = list(dict.fromkeys(row["Model"] for row in rows))
    conditions = list(dict.fromkeys(row["Condition"] for row in rows))
    values = {(row["Model"], row["Condition"]): row["Accuracy"] for row in rows}

    with (results_dir / "evaluation_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "Condition", "Accuracy"])
        writer.writeheader()
        writer.writerows(rows)

    with (results_dir / "accuracy_comparison.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Condition", *models])
        for condition in conditions:
            writer.writerow([
                condition, *(values[(model, condition)] for model in models)
            ])

    raw_conditions = [condition for condition in conditions if condition.startswith("Raw")]
    robustness = []
    for model in models:
        clean_accuracy = values[(model, "Clean")]
        corrupted_accuracy = sum(
            values[(model, condition)] for condition in raw_conditions
        ) / len(raw_conditions)
        robustness.append({
            "Model": model,
            "Clean Accuracy": clean_accuracy,
            "Average Corrupted Accuracy": corrupted_accuracy,
            "Accuracy Drop": clean_accuracy - corrupted_accuracy,
        })
    with (results_dir / "robustness_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(robustness[0]))
        writer.writeheader()
        writer.writerows(robustness)

    plot_grouped_accuracy(
        models, conditions, values,
        results_dir / "accuracy_comparison.png",
        "CNN and ViT Accuracy Under Different Conditions",
    )
    plot_grouped_accuracy(
        models, raw_conditions, values,
        results_dir / "augmentation_comparison.png",
        "Effect of Corruption-Augmented Training",
    )
    baseline_models = [name for name in models if name.startswith("baseline_")]
    filter_conditions = [
        condition for condition in conditions
        if condition.startswith(("Raw", "Strategy A"))
    ]
    plot_grouped_accuracy(
        baseline_models, filter_conditions, values,
        results_dir / "filtering_comparison.png",
        "Effect of Image Filtering on Baseline Models",
    )


def plot_grouped_accuracy(models, conditions, values, path, title):
    if not models or not conditions:
        return
    fig, ax = plt.subplots(figsize=(max(12, len(conditions) * 1.1), 7))
    width = 0.8 / len(models)
    positions = list(range(len(conditions)))
    for model_index, model in enumerate(models):
        offset = (model_index - (len(models) - 1) / 2) * width
        ax.bar(
            [position + offset for position in positions],
            [values[(model, condition)] for condition in conditions],
            width=width,
            label=model.replace("_", " ").title(),
        )
    ax.set_title(title)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Test Condition")
    ax.set_xticks(positions)
    ax.set_xticklabels(conditions, rotation=65, ha="right")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_cifar10_project(cfg, run, device, log, meta):
    if cfg["require_cuda"] and device.type != "cuda":
        raise RuntimeError(
            "This CIFAR-10 project config requires CUDA, but PyTorch cannot see a GPU."
        )
    valid_models = {
        "baseline_cnn", "baseline_vit", "augmented_cnn", "augmented_vit"
    }
    if (not isinstance(cfg["models"], list) or not cfg["models"]
            or set(cfg["models"]) - valid_models
            or len(set(cfg["models"])) != len(cfg["models"])):
        raise ValueError(f"models must be a non-empty list selected from {sorted(valid_models)}")
    if (cfg["batch_size"] <= 0 or cfg["patience"] <= 0
            or cfg["ckpt_every"] <= 0 or cfg["plot_every"] <= 0
            or cfg["num_workers"] < 0):
        raise ValueError(
            "batch_size, patience, ckpt_every, and plot_every must be positive; "
            "num_workers cannot be negative"
        )
    for name in cfg["models"]:
        epochs, learning_rate = cifar_training_settings(name, cfg)
        if epochs <= 0 or learning_rate <= 0:
            raise ValueError(f"{name} requires positive epochs and learning rate")

    train_subset, validation_subset, test_data = make_cifar_loaders(cfg)
    trained_models = {}
    model_histories = {}
    model_results = {}
    for name in cfg["models"]:
        torch.manual_seed(cfg["seed"])
        random.seed(cfg["seed"])
        if device.type == "cuda":
            torch.cuda.manual_seed_all(cfg["seed"])
        model, history, best_val_loss, best_epoch = train_cifar_model(
            name, train_subset, validation_subset, cfg, device, run, log, meta
        )
        trained_models[name] = model
        model_histories[name] = history
        model_results[name] = {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        }

    rows = evaluate_cifar_models(
        trained_models, test_data, cfg, device, run, log
    )
    for name in model_results:
        model_results[name]["clean_accuracy"] = next(
            row["Accuracy"] for row in rows
            if row["Model"] == name and row["Condition"] == "Clean"
        )
        raw_scores = [
            row["Accuracy"] for row in rows
            if row["Model"] == name and row["Condition"].startswith("Raw")
        ]
        model_results[name]["average_corrupted_accuracy"] = sum(raw_scores) / len(raw_scores)

    meta["model_results"] = model_results
    meta["model_histories"] = model_histories
    meta["epochs_run"] = sum(len(history["epoch"]) for history in model_histories.values())
    best_model = min(model_results, key=lambda name: model_results[name]["best_val_loss"])
    meta["best_model"] = best_model
    meta["best_epoch"] = model_results[best_model]["best_epoch"]
    meta["best_val_loss"] = model_results[best_model]["best_val_loss"]
    save_json(meta["model_histories"], run / "history.json")
    with (run / "results" / "training_history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "epoch", "loss", "accuracy", "val_loss", "val_accuracy"]
        )
        writer.writeheader()
        for name, history in model_histories.items():
            for index, epoch in enumerate(history["epoch"]):
                writer.writerow({
                    "model": name,
                    "epoch": epoch,
                    "loss": history["train_loss"][index],
                    "accuracy": history["train_accuracy"][index],
                    "val_loss": history["val_loss"][index],
                    "val_accuracy": history["val_accuracy"][index],
                })
    save_json(meta, run / "meta.json")
    plot_project_training_histories(model_histories, run / "plots")
    meta["status"] = "finished"
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(meta, run / "meta.json")
    log(f"Project results: {run / 'results'}")


def plot_project_training_histories(histories, plots_dir):
    plots_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, history in histories.items():
        label = name.replace("_", " ").title()
        axes[0].plot(history["epoch"], history["train_loss"], label=label)
        axes[0].plot(history["epoch"], history["val_loss"], linestyle="--")
        axes[1].plot(history["epoch"], history["train_accuracy"], label=label)
        axes[1].plot(history["epoch"], history["val_accuracy"], linestyle="--")
    axes[0].set_title("Training and validation loss")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[1].set_title("Training and validation accuracy")
    axes[1].set_ylabel("Accuracy (%)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "curves.png", dpi=150)
    plt.close(fig)


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
    if cfg["dataset"] == "cifar10" and cfg["require_cuda"] and device != "cuda":
        raise RuntimeError(
            "This CIFAR-10 project config requires CUDA, but PyTorch cannot see a GPU."
        )
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

    if cfg["dataset"] == "cifar10":
        run_cifar10_project(cfg, run, torch.device(device), log, meta)
        return
    if cfg["dataset"] != "synthetic":
        raise ValueError(f"Unknown dataset: {cfg['dataset']}")

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
