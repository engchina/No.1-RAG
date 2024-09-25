#!/bin/bash
# Source conda.sh to ensure conda command is available
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the desired conda environment
conda activate ginza-api

# Start a new shell session in the activated environment
PS1="(ginza-api) \u@\h:\w# " bash --noprofile --norc