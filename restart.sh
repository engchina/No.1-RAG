#!/bin/bash

PROJECT_DIR="/u01/aipoc/No.1-RAG"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/application_port.sh" || exit 1

lsof -ti:"$APPLICATION_PORT" | xargs kill -9 2>/dev/null || true
cd "$PROJECT_DIR" || exit 1
/bin/bash "$PROJECT_DIR/main.sh"
exit 0
