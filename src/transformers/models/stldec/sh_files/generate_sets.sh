#!/bin/bash
#SBATCH --qos=boost_qos_dbg
#SBATCH --no-requeue
#SBATCH --job-name="real"
#SBATCH --account IscrC_IRA-LLMs
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --mem=400G
#SBATCH --output=slurm_outputs/real.out

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

srun python3 generate_sets.py


echo "DONE!"

