#!/bin/bash

PROJECT_DIR="/u01/aipoc/No.1-RAG"

lsof -ti:8080 | xargs kill -9 2>/dev/null || true
cd "$PROJECT_DIR" || exit 1
/bin/bash "$PROJECT_DIR/main.sh"
exit 0
