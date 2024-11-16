#!/bin/bash

# EXAMPLE USAGE:
# sbatch fit_model_job.sh

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --job-name=lab4-fit-ensemble-models

#SBATCH --mail-user=bogdankostic@berkeley.edu
#SBATCH --mail-type=ALL

# Install AutoGluon
conda create -n autogluon python=3.11 -y  # Create a new conda environment, default python version on cluster is 3.12
conda activate autogluon
pip install -U pip
pip install -U setuptools wheel
pip install -U uv
uv pip install torch==2.3.1+cpu torchvision==0.18.1+cpu --index-url https://download.pytorch.org/whl/cpu
uv pip install autogluon

# Fit ensemble model on all features
python data_preprocessing.py > data_preprocessing_all.out
python fit_model.py > fit_model_all.out

# Fit ensemble model on engineered features only
python data_preprocessing.py --use_only_engineered_features > data_preprocessing_engineered.out
python fit_model.py --use_only_engineered_features > fit_model_job_engineered.out
