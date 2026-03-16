#!/bin/bash

# Usage: ./run_parallel_compilot.sh <num_workers> <screen_name_prefix> [config.yaml]
#
# Distributes the input programs across multiple workers (one per exclusive SLURM node)
# for parallel execution. Each worker processes a round-robin subset of programs and
# writes results to a worker-specific JSON file. Results are merged when all workers finish.

if [ $# -lt 2 ]; then
    echo "Usage: $0 <num_workers> <screen_name_prefix> [config.yaml]"
    exit 1
fi

NUM_WORKERS=$1
SCREEN_PREFIX=$2
CONFIG_FILE=${3:-config.yaml}

PYTHON_SCRIPT="compilot.py"
MERGE_SCRIPT="merge_results.py"

# Conda environment
CONDA_DIR=/share/apps/NYUAD5/miniconda/3-4.11.0
CONDA_ENV=/home/mm12191/.conda/envs/collm_env
CONDA_CMD=". $CONDA_DIR/bin/activate && conda activate $CONDA_ENV"

# SLURM settings (one exclusive node per worker for accurate timing measurements)
SBATCH_COMMAND="srun -J ${SCREEN_PREFIX} --exclude=dn105 --pty -n1 --exclusive --tasks-per-node=1 --cpus-per-task=28 --mem=100G -t 7-0:00:00 --constraint='dalma&avx2&compute&dalmac'"

echo "Launching $NUM_WORKERS parallel workers for $PYTHON_SCRIPT"
echo "Config: $CONFIG_FILE"
echo "Caution: Do not edit $PYTHON_SCRIPT while workers are running."
sleep 2

# Launch each worker in its own screen session
for WORKER_ID in $(seq 0 $((NUM_WORKERS - 1)))
do
    SCREEN_NAME="${SCREEN_PREFIX}_w${WORKER_ID}"
    COMPLETION_FILE="completed_${SCREEN_NAME}.flag"
    rm -f "$COMPLETION_FILE"

    screen -dmS "$SCREEN_NAME" bash -c "
        echo 'Starting worker $WORKER_ID / $NUM_WORKERS'
        while true; do
            $SBATCH_COMMAND bash -c '$CONDA_CMD && python $PYTHON_SCRIPT --worker-id=$WORKER_ID --num-workers=$NUM_WORKERS -c=$COMPLETION_FILE --config=$CONFIG_FILE'

            if [ -f \"$COMPLETION_FILE\" ]; then
                echo 'Worker $WORKER_ID completed successfully.'
                rm -f \"$COMPLETION_FILE\"
                break
            fi

            echo 'Worker $WORKER_ID terminated abnormally. Restarting in 5 seconds...'
            sleep 5
        done

        exec bash
    "

    echo "Started $SCREEN_NAME (worker $WORKER_ID/$NUM_WORKERS)"
done

echo ""
echo "All workers launched. Monitor with: screen -ls | grep $SCREEN_PREFIX"
echo "When all workers finish, merge results with:"
echo "  python $MERGE_SCRIPT --config $CONFIG_FILE [--num-workers $NUM_WORKERS]"
