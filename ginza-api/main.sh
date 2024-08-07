#!/bin/bash
cd /u01/aipoc/No.1-RAG/ginza-api && nohup /u01/aipoc/miniconda/envs/ginza-api/bin/python -m uvicorn main:app --host localhost --port 7932 &
exit 0