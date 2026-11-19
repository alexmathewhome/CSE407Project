#!/usr/bin/env bash
# Run one config with several seeds at the same time. No Slurm needed.
# Usage:  bash launch_seeds.sh CONFIG "SEEDS" [MAX_PARALLEL] [extra train.py arguments]
# Example: bash launch_seeds.sh configs/demo.yaml "0 1 2 3 4" 3
#          bash launch_seeds.sh configs/demo.yaml "0 1 2" 3 --set lr=0.001
# Activate the conda env first. Each seed writes to runs/<experiment>/seed_<n>/.
set -euo pipefail

CONFIG="${1:?Give a config file, for example configs/demo.yaml}"
SEEDS="${2:?Give the seeds in quotes, for example \"0 1 2\"}"
MAXP="${3:-3}"
shift 3 || shift $#
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" = "base" ]; then
    echo "Activate your env first:  conda activate cse407" >&2; exit 1
fi

EXP="$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['experiment'])" "$CONFIG")"
mkdir -p "runs/$EXP/_logs"
echo "Experiment $EXP | seeds: $SEEDS | at most $MAXP at once"

# xargs starts at most MAXP processes. When one ends, the next seed starts.
printf '%s\n' $SEEDS | xargs -P "$MAXP" -I{} bash -c \
    'python -u train.py --config "$0" --seed {} "${@:2}" > "runs/$1/_logs/seed_{}.out" 2>&1 \
     && echo "seed {} done" || echo "seed {} FAILED (see runs/$1/_logs/seed_{}.out)"' \
    "$CONFIG" "$EXP" "$@"

python summarize.py "$EXP"
