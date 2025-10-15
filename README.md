# Aim for No.1 RAG

## Deploy


- v1.1.9(Pre-Release): 東京/大阪/シカゴリージョンのみをサポートしています。（デフォルト：大阪リージョン）
  - fix: embed_id start from 10001

  Click [![Deploy to Oracle Cloud](https://oci-resourcemanager-plugin.plugins.oci.oraclecloud.com/latest/deploy-to-oracle-cloud.svg)](https://cloud.oracle.com/resourcemanager/stacks/create?region=ap-osaka-1&zipUrl=https://github.com/engchina/No.1-RAG/releases/download/v1.1.9/v1.1.9.zip)


- v1.1.8(Pre-Release): 東京/大阪/シカゴリージョンのみをサポートしています。（デフォルト：大阪リージョン）
  - fix: Add UTF-8 encoding note for .txt/.md files

  Click [![Deploy to Oracle Cloud](https://oci-resourcemanager-plugin.plugins.oci.oraclecloud.com/latest/deploy-to-oracle-cloud.svg)](https://cloud.oracle.com/resourcemanager/stacks/create?region=ap-osaka-1&zipUrl=https://github.com/engchina/No.1-RAG/releases/download/v1.1.8/v1.1.8.zip)

- v1.1.7(Pre-Release): 東京/大阪/シカゴリージョンのみをサポートしています。（デフォルト：大阪リージョン）
  - modify: load .txt file by TextLoader
  - modify: handle json parser error

- v1.1.5(Pre-Release): 東京/大阪/シカゴリージョンのみをサポートしています。（デフォルト：大阪リージョン）
  - fix: ORA-64451: failed to escape special character 0x1

- v1.1.4(Pre-Release): 東京/大阪/シカゴリージョンのみをサポートしています。（デフォルト：大阪リージョン）
  - add model: xai.grok-4
  - remove model: xai.grok-3
  - remove model: meta.llama-4-maverick-17b-128e-instruct-fp8
  - remove model: meta.llama-3-3-70b
  - remove model: meta.llama-3-2-90b-vision
  - remove model: openai/gpt-4o
  - remove model: azure_openai/gpt-4o

- v1.1.1: 東京/大阪/シカゴリージョンのみをサポートしています。（デフォルト：大阪リージョン）
  - update: oracle instance client 
  
  


- v1.1.0(Pre-Release): 東京/大阪/シカゴリージョンのみをサポートしています。（デフォルト：大阪リージョン）
  - add model: xai.grok-3
  - add model: cohere.embed-v4.0
  - remove model: cohere.embed-multilingual-v3.0
  - remove model: claude-opus, claude-sonnet, claude-haiku
  - add feature: Vision 回答
  - add feature: Pdf2Markdown (only for VQA)
  - remove feature: Markitdown 
  
 
- v1.0.9: 東京/大阪/シカゴリージョンのみをサポートしています。（デフォルト：大阪リージョン）
  - support meta.llama-4-maverick-17b-128e-instruct-fp8
  - support meta.llama-4-scout-17b-16e-instruct
  - support modify rag prompt through ui

## Local Development Setup (No need for deploy to OCI)

```
conda create -n no.1-rag python=3.11 -y
conda activate no.1-rag
```

```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install gradio==5.18.0
# unstructured-client 0.30.4 requires aiofiles>=24.1.0, but you have aiofiles 23.2.1 which is incompatible.
pip install aiofiles==24.1.0
# gradio 5.18.0 requires aiofiles<24.0,>=22.0, but you have aiofiles 24.1.0 which is incompatible.
pip install defusedxml==0.7.1 markitdown==0.0.2 pathvalidate==3.2.3 speechrecognition==3.14.1 youtube-transcript-api==1.0.1
# pip list --format=freeze | grep -iv gradio > requirements.txt
```

```
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

## Other License

- markitdown package is released under the [MIT license](https://github.com/microsoft/markitdown).