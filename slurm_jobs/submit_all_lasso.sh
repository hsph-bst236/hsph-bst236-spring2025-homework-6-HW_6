#!/bin/bash
# Command to submit this script: ./slurm_jobs/submit_all_lasso.sh
# Define the optimizers to test
OPTIMIZERS=("GD" "SGD" "Adam" "ISTA" "FISTA")
# Define the learning rates to test
LEARNING_RATES=(0.1 0.01 0.001 0.0001)

echo "Submitting jobs ..."
echo "==============================================================="

mkdir -p slurm_logs
# Count submitted jobs
job_count=0

# Loop through each optimizer and learning rate
for optimizer in "${OPTIMIZERS[@]}"; do   
    for lr in "${LEARNING_RATES[@]}"; do
        echo "  Submitting job with Optimizer: $optimizer and learning rate: $lr"
        
        # Submit individual job for this optimizer and learning rate
        job_id=$(sbatch --parsable ./slurm_jobs/run_single_lasso_job.sh "$optimizer" "$lr")
        job_count=$((job_count + 1))
    done
done

echo "==============================================================="
echo "All jobs submitted: $job_count total jobs"