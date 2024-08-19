#!/bin/bash
cd /u01/aipoc/No.1-RAG/ginza-api && nohup /u01/aipoc/miniconda/envs/ginza-api/bin/python -m uvicorn main:app --host localhost --port 7932 > /u01/aipoc/No.1-RAG/ginza.log 2>&1 &
 &
exit 0