#!/bin/bash
# Parse command line arguments
OPTIMIZER=$1
LEARNING_RATE=$2

#SBATCH --job-name=lasso_lr
#SBATCH --output=slurm_logs/lasso_${OPTIMIZER}_lr${LEARNING_RATE}_%j.out
#SBATCH --error=slurm_logs/lasso_${OPTIMIZER}_lr${LEARNING_RATE}_%j.err
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#
# Script to run a single Lasso regression job with specific optimizer and learning rate
# Usage: sbatch run_single_lasso_job.sh <optimizer> <learning_rate>



# Create the log directory if it doesn't exist
mkdir -p slurm_logs

# Print job information
echo "Job started: Running with optimizer $OPTIMIZER and learning rate $LEARNING_RATE"

# activate venv
source venv/bin/activate

# Run the main_Lasso.py script with the provided parameters
python3 src/main_Lasso.py --optimizer $OPTIMIZER --learning_rate $LEARNING_RATE