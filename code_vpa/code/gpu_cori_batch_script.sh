#!/bin/bash
#################
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --qos=regular
#SBATCH --job-name=ice_cube_cnn_train
#SBATCH --output=slurm-%x-%j.out
#SBATCH --constraint=gpu
#SBATCH --account=nstaff
#################


echo "--start date" `date` `date +%s`
echo '--hostname ' $HOSTNAME
export HDF5_USE_FILE_LOCKING=FALSE
module unload esslurm
module load python 
conda activate v_py3
### Actual script to run
python main.py --config config_cori.yaml --gpu cori --model_list $1
echo "--end date" `date` `date +%s`
