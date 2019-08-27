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
# Limit to one GPU
#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=2
### Actual script to run
#echo $2
python main.py --config config_cori.yaml --gpu cori --model_list 1 2
#python3 Cnn_train.py --typeofdata $1 --model_list $2
echo "--end date" `date` `date +%s`
