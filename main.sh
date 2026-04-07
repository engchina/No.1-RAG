#!/bin/bash

PROJECT_DIR="/u01/aipoc/No.1-RAG"

cd "$PROJECT_DIR" || exit 1
nohup /u01/aipoc/miniconda/envs/no.1-rag/bin/python main.py --host 0.0.0.0 > /u01/aipoc/No.1-RAG/main.log 2>&1 &
exit 0
