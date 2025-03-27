#!/bin/bash
echo "Initializing application setup..."

# Git clone source code
cd /u01/aipoc/No.1-RAG
dos2unix main.cron
crontab main.cron

# Docker setup
chmod +x ./langfuse/install_docker.sh
./langfuse/install_docker.sh
systemctl start docker

docker network create aipoc-network

# Dowanload and configure instantclient
cd /u01/aipoc
wget https://download.oracle.com/otn_software/linux/instantclient/2350000/instantclient-basic-linux.x64-23.5.0.24.07.zip -O /u01/aipoc/instantclient-basic-linux.x64-23.5.0.24.07.zip
unzip /u01/aipoc/instantclient-basic-linux.x64-23.5.0.24.07.zip -d ./
wget http://ftp.de.debian.org/debian/pool/main/liba/libaio/libaio1_0.3.113-4_amd64.deb
dpkg -i libaio1_0.3.113-4_amd64.deb
sh -c "echo /u01/aipoc/instantclient_23_5 > /etc/ld.so.conf.d/oracle-instantclient.conf"
ldconfig
echo 'export LD_LIBRARY_PATH=/u01/aipoc/instantclient_23_5:$LD_LIBRARY_PATH' >> /etc/profile
source /etc/profile

# Unzip wallet and copy essential file to instantclient
unzip /u01/aipoc/wallet.zip -d ./wallet
cp /u01/aipoc/wallet/*  /u01/aipoc/instantclient_23_5/network/admin/

# Application setup
EXTERNAL_IP=$(curl -s -m 10 http://whatismyip.akamai.com/)
sed -i "s|localhost:3000|$EXTERNAL_IP:3000|g" ./langfuse/docker-compose.yml
chmod +x ./langfuse/main.sh
nohup ./langfuse/main.sh &

# Update environment variables
DB_CONNECTION_STRING=$(cat /u01/aipoc/props/db.env)
COMPARTMENT_ID=$(cat /u01/aipoc/props/compartment_id.txt)
cp .env.example .env
DB_CONNECTION_STRING=$(cat /u01/aipoc/props/db.env)
sed -i "s|ORACLE_23AI_CONNECTION_STRING=TODO|ORACLE_23AI_CONNECTION_STRING=$DB_CONNECTION_STRING|g" .env
COMPARTMENT_ID=$(cat /u01/aipoc/props/compartment_id.txt)
sed -i "s|OCI_COMPARTMENT_OCID=TODO|OCI_COMPARTMENT_OCID=$COMPARTMENT_ID|g" .env

# Install and configure miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /u01/aipoc/miniconda.sh
bash miniconda.sh -b -p /u01/aipoc/miniconda
eval "$(/u01/aipoc/miniconda/bin/conda shell.bash hook)"
/u01/aipoc/miniconda/bin/conda init bash

# Run ginza-api
conda create -n ginza-api python=3.11 -y
conda activate ginza-api
pip install -r ./ginza-api/requirements.txt
chmod +x ./ginza-api/main.sh
nohup ./ginza-api/main.sh &

# Run application
conda create -n no.1-rag python=3.11 -y
conda activate no.1-rag
pip install -r requirements.txt
pip install gradio==5.18.0
pip install aiofiles==24.1.0
pip install defusedxml==0.7.1 markitdown==0.0.2 pathvalidate==3.2.3 speechrecognition==3.14.1 youtube-transcript-api==1.0.1
python -m nltk.downloader -u https://pastebin.com/raw/D3TBY4Mj punkt averaged_perceptron_tagger

chmod +x main.sh
nohup ./main.sh &

echo "Initialization complete."
