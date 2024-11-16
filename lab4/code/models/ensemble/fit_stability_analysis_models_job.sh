#!/bin/bash

# EXAMPLE USAGE:
# sbatch fit_model_job.sh <test_image>

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --job-name=lab4-fit-stability-analysis-models

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

# Fit models for stability analysis
python fit_stability_analysis_models.py $1 > fit_stability_analysis_models_$1.out
