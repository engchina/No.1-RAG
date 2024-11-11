# Aim for No. 1 RAG

## Deploy

Click [![Deploy to Oracle Cloud](https://oci-resourcemanager-plugin.plugins.oci.oraclecloud.com/latest/deploy-to-oracle-cloud.svg)](https://cloud.oracle.com/resourcemanager/stacks/create?region=ap-tokyo-1&zipUrl=https://github.com/engchina/No.1-RAG/releases/download/no.1_rag_v1.0.0/v1.0.0.zip)

## Setup

```
conda create -n no.1-rag python=3.11 -y
conda activate no.1-rag
```

```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip list --format=freeze > requirements.txt
```

```angular2html
# Test cuda
python -c "import torch;print(torch.cuda.is_available())"

# Expected
True
```

## Full Text Search

```
BEGIN
  CTX_DDL.CREATE_PREFERENCE('world_lexer','WORLD_LEXER');
END;
/

CREATE INDEX embed_data_idx ON langchain_oracle_embedding(embed_data) INDEXTYPE IS CTXSYS.CONTEXT PARAMETERS ('LEXER world_lexer sync (on commit)');
```

## Using Unstructured Open Source

```
sudo apt install build-essential libmagic-dev poppler-utils tesseract-ocr libreoffice pandoc -y
```

## Generate LangFuse Secret and Salt

```
echo <your_secret> | openssl rand -base64 32
echo <your_salt> | openssl rand -base64 32
```

## Change Region Image ID

[https://docs.oracle.com/en-us/iaas/images/image/50cf60da-4374-44e2-ab38-70185991f833/index.htm](https://docs.oracle.com/en-us/iaas/images/image/50cf60da-4374-44e2-ab38-70185991f833/index.htm)