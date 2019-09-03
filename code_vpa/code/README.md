


## Running the Python code
- Default way (train and test model)
` python main.py --train --config config_cori.yaml --gpu cori --model_list 1 2 `

- Only test models
` python main.py --config config_cori.yaml --gpu cori --model_list 1 2  `


## Running the submit script
- Run models 1,2,3 on cori GPU
` sbatch gpu_cori_batch_script.sh 1\ 2\ 3 `
- Run models 1,2,3 on maeve GPU
` sbatch gpu_maeve_batch_script.sh 1\ 2\ 3 `


