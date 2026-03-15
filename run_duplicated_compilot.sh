#!/bin/bash

# Usage: ./run_duplicated_compilot.sh <num_runs> <screen_name_prefix> [script_args]

# Check for required arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <num_runs> <screen_name_prefix>"
    exit 1
fi

# Assign arguments to variables
NUM_RUNS=$1            # Number of duplicates to start
SCREEN_PREFIX=$2       # Screen session prefix

PYTHON_SCRIPT="compilot.py"    # The Python script to run

# The name of your Conda environment
CONDA_CMD="source /data/scratch/mmerouani/anaconda/bin/activate"
# SLURM settings
SBATCH_COMMAND="srun -J ${SCREEN_PREFIX} --exclude lanka20,lanka04,lanka21,lanka03,lanka09 -p lanka-v2 -q commit-main --exclusive -N 1 --mem=120gb -t 3-10:00:00"

echo "Caution⚠: This script will rerun $PYTHON_SCRIPT indefinitely until it terminates properly. Make sure to not edit $PYTHON_SCRIPT while this script is running to avoid erroneous results."
sleep 2

# Loop to create screen sessions
for i in $(seq 1 $NUM_RUNS)
do
    # Create a unique screen session name
    SCREEN_NAME="${SCREEN_PREFIX}_${i}"
    COMPLETION_FILE="completed_${SCREEN_NAME}.flag"
    rm -f $COMPLETION_FILE  # Ensure old completion file is removed before starting

    # Start a new screen session, activate the Conda environment, and run the Python script
    screen -dmS $SCREEN_NAME bash -c "
        echo 'Starting experiment'
        while true; do
            
            $SBATCH_COMMAND bash -c '$CONDA_CMD && python $PYTHON_SCRIPT -i=$i -c=$COMPLETION_FILE'
            # Check if the success file was created
            if [ -f \"$COMPLETION_FILE\" ]; then
                echo 'Experiment completed successfully.'
                rm -f \"$COMPLETION_FILE\"  # Cleanup after success
                break
            fi
            echo 'file \"$COMPLETION_FILE\" not found'
            
            echo 'Experiment terminated abnormally. Restarting in 5 seconds...'
            sleep 5
        done
        
        exec bash
    "
    
    echo "Started $SCREEN_NAME running: python $PYTHON_SCRIPT $i"
done


