#!/bin/bash
# Source conda.sh to ensure conda command is available
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the desired conda environment
conda activate no.1-rag

# Start a new shell session in the activated environment
PS1="(no.1-rag) \u@\h:\w# " bash --noprofile --norc