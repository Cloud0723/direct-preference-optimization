#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=62gb
#SBATCH --output=log/%j.out                              
#SBATCH --error=log/%j.out
#SBATCH --job-name=dpo
#SBATCH -n 12
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH -p a100-4

source activate dpo
nvidia-smi

cd ~/jobsubmit/direct-preference-optimization || exit
python -u train.py model=pythia69 datasets=[hh] loss=sft exp_name=anthropic_dpo_pythia69 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false

exit