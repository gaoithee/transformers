#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="eval"
#SBATCH --account IscrC_IRA-LLMs
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --mem=481G
#SBATCH --output=slurm_outputs/validate_step_24500.out

module load python/3.11.6--gcc--8.5.0

echo "Running on $SLURM__NNODES nodes"

# Standard preamble for debugging
echo "---------------------------------------------"
echo "SLURM job ID:        $SLURM_JOB_ID"
echo "SLURM job node list: $SLURM_JOB_NODELIST"
echo "DATE:                $(date)"
echo "---------------------------------------------"


source /leonardo/home/userexternal/scanduss/.venv/bin/activate


# Needed exports
# export <export_name>=<export_value>
#variables

srun python3 validate.py


echo "DONE!"

