#!/usr/bin/env bash
# One experiment end to end: precompute, train, test.
#
#   ./reproduce.sh <experiment> <model_type> [dataset_size] [log_id]
#   ./reproduce.sh wind->wind S3CNNUNet 1 paper
#   ./reproduce.sh MNIST-scalar-classification S3CNNMNIST
#
# The pair must be one the registry knows; `python -c 'from modules import zoo;
# print(sorted(zoo.ZOO))'` lists them. Both jobs run through
# run_all_experiments.py, so exactly one idle GPU is claimed for the pair and
# held until they finish.
set -euo pipefail

EXPERIMENT=${1:?usage: reproduce.sh <experiment> <model_type> [dataset_size] [log_id]}
MODEL=${2:?usage: reproduce.sh <experiment> <model_type> [dataset_size] [log_id]}
SIZE=${3:-1}
LOG_ID=${4:-reproduce}
EPOCHS=${EPOCHS:-100}
BATCH=${BATCH:-1}

# The registry knows the grid and the results directory; ask it rather than
# repeating them here.
read -r N_BETA MODE IS_WEATHER <<EOF
$(python - "$EXPERIMENT" "$MODEL" <<'PY'
import sys
from modules import zoo
experiment, model_type = sys.argv[1], sys.argv[2]
spec, task = zoo.spec(experiment, model_type), zoo.task(experiment)
print(task.n_beta, spec.mode, int(task.data == "weather"))
PY
)
EOF

echo "== precompute (n_beta=$N_BETA)"
python generate_precomps.py --n_beta "$N_BETA" --l_max "$N_BETA" --compute_wigner

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

python - "$EXPERIMENT" "$MODEL" "$SIZE" "$LOG_ID" "$EPOCHS" "$BATCH" "$IS_WEATHER" \
        > "$WORK/reproduce_schedule.py" <<'PY'
import sys
from run_training import get_experiment_path
from modules import zoo

experiment, model, size, log_id, epochs, batch, weather = sys.argv[1:8]
size = int(size) if int(weather) else None
mode = zoo.mode(experiment, model)
path = get_experiment_path(experiment, mode, "no", size, log_id)

train = dict(operation="train", experiment=experiment, model_type=model, rotate="no",
             total_epochs=int(epochs), batch_size=int(batch), log_id=log_id,
             dataset_size=size)
test = dict(operation="test", experiment=experiment, model_type=model, rotate="no",
            batch_size=int(batch), path=path)
print("def create_schedule():")
print(f"    return [{train!r}, {test!r}]")
PY

echo "== train and test ($MODE, results in experiments/$EXPERIMENT/$MODE/NR/)"
PYTHONPATH="$WORK:${PYTHONPATH:-}" python run_all_experiments.py --schedule reproduce_schedule

echo "== done; loss curves and checkpoints are under"
python - "$EXPERIMENT" "$MODEL" "$SIZE" "$LOG_ID" "$IS_WEATHER" <<'PY'
import sys
from run_training import get_experiment_path
from modules import zoo
experiment, model, size, log_id, weather = sys.argv[1:6]
size = int(size) if int(weather) else None
print("  " + get_experiment_path(experiment, zoo.mode(experiment, model), "no", size, log_id))
PY
echo "Figures: the scripts in plotting/ read run ids; add this one to the list there."
