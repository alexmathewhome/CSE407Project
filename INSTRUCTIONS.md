# Running your project on the Spark

## What is in this folder

| File | Purpose |
|---|---|
| `setup_env.sh` | Checks Python, installs conda if needed, makes the env, installs PyTorch and the other packages, runs a short test. |
| `requirements.txt` | Packages other than PyTorch. Add your own packages here. |
| `train.py` | Training template. Replace the data and the model. Keep the rest. |
| `configs/*.yaml` | One file per experiment. These hold all hyperparameters. |
| `launch_seeds.sh` | Runs one config with several seeds at the same time. |
| `summarize.py` | Prints a table of all seeds of one experiment. |
| `runs/` | All output. Git ignores this folder. |

## 1. Get the code

```bash
git clone <repo-url>
cd <repo-name>
```

## 2. Set up the environment (once)

```bash
bash setup_env.sh
```

The script installs everything and then runs a 5-epoch test.
The test is correct when the last lines list these files under `runs/smoke/seed_0/`:

```
checkpoints/best.pt
checkpoints/last.pt
config.yaml
history.json
meta.json
plots/curves.png
train.log
```

The script prints `cuda usable : True` when PyTorch can use the GPU.
If it prints `False`, stop and tell the instructor.

Activate the env in every new terminal:

```bash
conda activate cse407
```

If `conda activate` fails, run `conda init bash` once and open a new terminal.

To use another env name or Python version:

```bash
bash setup_env.sh myenv 3.12
```

To add a package later:

```bash
conda activate cse407
pip install <package>
```

Then add the package name to `requirements.txt`.

## 3. Run one training job

```bash
python train.py --config configs/demo.yaml
```

Change one setting without editing the file:

```bash
python train.py --config configs/demo.yaml --seed 3 --set lr=0.001 --set patience=5
```

## 4. Where the output goes

Each run has its own folder: `runs/<experiment>/seed_<seed>/`.

| Path | Content |
|---|---|
| `config.yaml` | The exact settings of this run, including the values you changed with `--set`. |
| `meta.json` | Status, start and end time, host, Python and PyTorch version, GPU name, git commit, best epoch, best validation loss. |
| `history.json` | Training loss and validation loss for each epoch. |
| `train.log` | The printed output. |
| `checkpoints/last.pt` | Model, optimizer, epoch, early-stop counter, history. Saved every `ckpt_every` epochs. Used to resume. |
| `checkpoints/best.pt` | Model weights from the epoch with the lowest validation loss. Use this file for your final results. |
| `plots/curves.png` | Training and validation loss. Redrawn every `plot_every` epochs and at the end. |

`status` in `meta.json` is one of `running`, `finished`, `early_stopped`.
A run that stopped without reaching `finished` or `early_stopped` was interrupted.

## 5. Resume an interrupted run

Run the same command again.
`train.py` loads `checkpoints/last.pt` and continues from the next epoch.

To start again from epoch 0, delete the run folder first:

```bash
rm -rf runs/demo/seed_0
```

Resuming restores the model, the optimizer and the early-stop counter.
It does not restore the random number state, so the batch order after a resume differs from an uninterrupted run.

## 6. Settings (config files)

Copy `configs/demo.yaml` to `configs/<my_experiment>.yaml`.
Change `experiment` to a new name.
Each experiment name gets its own folder under `runs/`.

| Key | Meaning |
|---|---|
| `experiment` | Folder name under `runs/`. |
| `seed` | Sets model initialisation and batch order. |
| `data_seed` | Sets data generation. Keep it fixed when you compare seeds. |
| `epochs` | Maximum number of epochs. |
| `batch_size` | Samples per update. |
| `lr` | Learning rate. |
| `hidden` | Hidden layer width. |
| `patience` | Number of epochs without improvement before early stopping. |
| `min_delta` | Smallest decrease in validation loss that counts as an improvement. |
| `ckpt_every` | Epochs between saves of `last.pt`. |
| `plot_every` | Epochs between saves of `curves.png`. |

To add a setting, add it to `DEFAULTS` in `train.py`, then use `cfg["name"]` in the code.
`train.py` stops with an error if a config file has a key that is not in `DEFAULTS`.

## 7. Run several seeds at the same time

The Spark has no Slurm. `launch_seeds.sh` starts one process per seed.

```bash
conda activate cse407
bash launch_seeds.sh configs/demo.yaml "0 1 2 3 4" 3
```

The arguments are the config file, the seeds in quotes, and the maximum number of runs at one time.
In this example, 3 runs start. When one ends, the next seed starts.
All runs share the one GPU.
Start with 3 at one time.
Increase the number only if `nvidia-smi` shows free GPU memory and the runs do not slow down.

Extra arguments go to `train.py`:

```bash
bash launch_seeds.sh configs/demo.yaml "0 1 2" 3 --set lr=0.001
```

Each seed prints to `runs/<experiment>/_logs/seed_<n>.out`.
When all seeds end, the script prints a table with the mean and standard deviation of the best validation loss.
To print the table again:

```bash
python summarize.py demo
```

## 8. Keep a job running after you log out

A job started in a plain terminal stops when the connection closes.
Use `tmux`:

```bash
tmux new -s train
conda activate cse407
bash launch_seeds.sh configs/demo.yaml "0 1 2 3 4" 3
```

Press `Ctrl-b`, then `d`, to leave the session. The job continues.
To return:

```bash
tmux attach -t train
```

Use `nohup` if `tmux` is not available:

```bash
nohup bash launch_seeds.sh configs/demo.yaml "0 1 2 3 4" 3 > launch.out 2>&1 &
```

## 9. Check a running job

| Command | Shows |
|---|---|
| `nvidia-smi` | GPU use and the processes on the GPU. |
| `tail -f runs/demo/seed_0/train.log` | Live output of one run. Press `Ctrl-c` to stop viewing. |
| `python summarize.py demo` | Status and best result of every seed. |
| `ps aux \| grep train.py` | The running training processes. |

To stop one run, find its process ID with `ps` and run `kill <pid>`.
The run can resume later.

## 10. Use your own data and model

Edit the block in `train.py` between the two comment lines `Replace this block` and the dashed line.
Keep these parts unchanged:

- The `cfg` settings.
- The saving of `last.pt` and `best.pt`.
- The early-stopping check.
- The writing of `history.json` and `meta.json`.

Do not commit `runs/` or `.pt` files to git. `.gitignore` already excludes them.
Copy `best.pt`, `config.yaml` and `plots/curves.png` out of the run folder for your report.

## Common problems

| Message | Action |
|---|---|
| `Activate your env first` | Run `conda activate cse407`. |
| `Unknown config keys` | A key in the YAML file is misspelled or missing from `DEFAULTS` in `train.py`. |
| `CUDA out of memory` | Lower `batch_size` or run fewer seeds at the same time. |
| `cuda usable : False` | Tell the instructor. Do not train on the CPU. |
