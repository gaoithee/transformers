#!/bin/bash
#SBATCH --no-requeue
#SBATCH --account=dssc
#SBATCH --job-name="test16"
#SBATCH --get-user-env
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1                  # <-- TODO adjust this
#SBATCH --mem=50G                             # <-- TODO adjust this
#SBATCH --time=04:00:00                       # <-- TODO adjust this
#SBATCH --output=slurm_outs/correct_training_16.out


# Standard preamble for debugging
echo "---------------------------------------------"
echo "SLURM job ID:        $SLURM_JOB_ID"
echo "SLURM job node list: $SLURM_JOB_NODELIST"
echo "DATE:                $(date)"
echo "---------------------------------------------"


# Needed sourcing
source .venv/bin/activate

# Needed modules
# module load <module_name>

# Needed exports
# export <export_name>=<export_value>
#variables
FILE_NAME=/u/dssc/scandu00/transformers/src/transformers/models/stldec/train-16batch.py

CMD="python3 -u"

if [ ! -f "$FILE_NAME" ]; then
  echo "The file $FILE_NAME does not exist"
  exit 1
fi

# Other checks there

$CMD $FILE_NAME

echo "DONE!"
