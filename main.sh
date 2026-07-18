#!/bin/bash

PROJECT_DIR="/u01/aipoc/No.1-RAG"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/application_port.sh" || exit 1

cd "$PROJECT_DIR" || exit 1
nohup /u01/aipoc/miniconda/envs/no.1-rag/bin/python main.py --host 0.0.0.0 --port "$APPLICATION_PORT" > /u01/aipoc/No.1-RAG/main.log 2>&1 &
exit 0
