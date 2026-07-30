#!/bin/bash
set -euo pipefail

exec > >(tee -a /var/log/init_script.log) 2>&1

echo "Initializing No.1-RAG application setup..."

INSTALL_DIR="/u01/aipoc"
PROJECT_DIR="${INSTALL_DIR}/No.1-RAG"
INSTANTCLIENT_VERSION="23.26.0.0.0"
INSTANTCLIENT_ZIP="instantclient-basic-linux.x64-${INSTANTCLIENT_VERSION}.zip"
INSTANTCLIENT_URL="https://download.oracle.com/otn_software/linux/instantclient/2326000/${INSTANTCLIENT_ZIP}"
INSTANTCLIENT_SQLPLUS_ZIP="instantclient-sqlplus-linux.x64-${INSTANTCLIENT_VERSION}.zip"
INSTANTCLIENT_SQLPLUS_URL="https://download.oracle.com/otn_software/linux/instantclient/2326000/${INSTANTCLIENT_SQLPLUS_ZIP}"
LIBAIO_DEB="libaio1_0.3.113-4_amd64.deb"
LIBAIO_URL="http://ftp.de.debian.org/debian/pool/main/liba/libaio/${LIBAIO_DEB}"
INSTANTCLIENT_DIR="${INSTALL_DIR}/instantclient_23_26"

source "${PROJECT_DIR}/install_utils.sh"

retry_command() {
    local max_attempts=5
    local timeout=10
    local attempt=1
    local exit_code=0

    while [ "$attempt" -le "$max_attempts" ]; do
        echo "Attempt ${attempt}/${max_attempts}: $*"
        "$@" && return 0
        exit_code=$?
        echo "Command failed with exit code ${exit_code}. Retrying in ${timeout} seconds..."
        sleep "$timeout"
        attempt=$((attempt + 1))
        timeout=$((timeout * 2))
    done

    echo "Command failed after ${max_attempts} attempts."
    return "$exit_code"
}

conda_env_exists() {
    conda env list | awk '{print $1}' | grep -qx "$1"
}

set_env_value() {
    local key="$1"
    local value="$2"
    local escaped_value
    escaped_value=$(printf '%s' "$value" | sed -e 's/[\/&|]/\\&/g')
    if grep -q "^${key}=" .env; then
        sed -i "s|^${key}=.*|${key}=${escaped_value}|g" .env
    else
        printf '%s=%s\n' "$key" "$value" >> .env
    fi
}

cd "$INSTALL_DIR"

echo "Installing Miniconda..."
if [ ! -d "${INSTALL_DIR}/miniconda" ]; then
    retry_command wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "${INSTALL_DIR}/miniconda.sh"
    bash "${INSTALL_DIR}/miniconda.sh" -b -p "${INSTALL_DIR}/miniconda"
    eval "$("${INSTALL_DIR}/miniconda/bin/conda" shell.bash hook)"
    "${INSTALL_DIR}/miniconda/bin/conda" init bash
else
    echo "Miniconda is already installed."
    eval "$("${INSTALL_DIR}/miniconda/bin/conda" shell.bash hook)"
fi

echo "Installing Oracle Instant Client ${INSTANTCLIENT_VERSION}..."
if [ ! -d "${INSTANTCLIENT_DIR}" ]; then
    if [ ! -f "${INSTANTCLIENT_ZIP}" ]; then
        retry_command wget "${INSTANTCLIENT_URL}" -O "${INSTANTCLIENT_ZIP}"
    fi
    unzip -o "${INSTANTCLIENT_ZIP}" -d ./

    if [ ! -f "${INSTANTCLIENT_SQLPLUS_ZIP}" ]; then
        retry_command wget "${INSTANTCLIENT_SQLPLUS_URL}" -O "${INSTANTCLIENT_SQLPLUS_ZIP}"
    fi
    unzip -o "${INSTANTCLIENT_SQLPLUS_ZIP}" -d ./

    if [ ! -f "${LIBAIO_DEB}" ]; then
        retry_command wget "${LIBAIO_URL}" -O "${LIBAIO_DEB}"
    fi
    dpkg -i "${LIBAIO_DEB}" || apt_get_with_retry install -f -y

    sh -c "echo ${INSTANTCLIENT_DIR} > /etc/ld.so.conf.d/oracle-instantclient.conf"
    ldconfig

    if ! grep -q "LD_LIBRARY_PATH=${INSTANTCLIENT_DIR}" /etc/profile; then
        echo "export LD_LIBRARY_PATH=${INSTANTCLIENT_DIR}:\$LD_LIBRARY_PATH" >> /etc/profile
        echo "export PATH=${INSTANTCLIENT_DIR}:\$PATH" >> /etc/profile
    fi
else
    echo "Oracle Instant Client is already installed."
fi

mkdir -p "${INSTANTCLIENT_DIR}/network/admin"

set +eu
source /etc/profile
set -eu
export LD_LIBRARY_PATH="${INSTANTCLIENT_DIR}:${LD_LIBRARY_PATH:-}"
export PATH="${INSTANTCLIENT_DIR}:$PATH"

if command -v sqlplus >/dev/null 2>&1; then
    echo "SQL*Plus installation verified."
else
    echo "Error: SQL*Plus installation verification failed."
    exit 1
fi

echo "Configuring No.1-RAG project..."
cd "$PROJECT_DIR"

dos2unix main.cron
crontab main.cron

echo "Configuring environment variables..."
cp .env.example .env

if [ -f "/u01/aipoc/props/db.env" ]; then
    DB_CONNECTION_STRING=$(cat /u01/aipoc/props/db.env)
    set_env_value "ORACLE_23AI_CONNECTION_STRING" "$DB_CONNECTION_STRING"
else
    echo "Warning: /u01/aipoc/props/db.env was not found."
fi

if [ -f "/u01/aipoc/props/compartment_id.txt" ]; then
    COMPARTMENT_ID=$(cat /u01/aipoc/props/compartment_id.txt)
    set_env_value "OCI_COMPARTMENT_OCID" "$COMPARTMENT_ID"
else
    echo "Warning: /u01/aipoc/props/compartment_id.txt was not found."
fi

ADB_NAME=$(cat /u01/aipoc/props/adb_name.txt 2>/dev/null || true)
if [ -n "$ADB_NAME" ]; then
    set_env_value "ADB_NAME" "$ADB_NAME"
fi

if [ -f "/u01/aipoc/props/adb_ocid.txt" ]; then
    ADB_OCID=$(cat /u01/aipoc/props/adb_ocid.txt)
    set_env_value "ADB_OCID" "$ADB_OCID"
else
    echo "Warning: /u01/aipoc/props/adb_ocid.txt was not found."
fi

set_env_value "ORACLE_CLIENT_LIB_DIR" "$INSTANTCLIENT_DIR"

echo "Configuring Autonomous Database wallet..."
WALLET_SRC=""
if [ -f "/u01/aipoc/props/wallet.zip" ]; then
    WALLET_SRC="/u01/aipoc/props/wallet.zip"
elif [ -f "/u01/aipoc/wallet.zip" ]; then
    WALLET_SRC="/u01/aipoc/wallet.zip"
fi

if [ -n "$WALLET_SRC" ]; then
    WALLET_DIR="${INSTANTCLIENT_DIR}/network/admin"
    mkdir -p "$WALLET_DIR"
    unzip -o "$WALLET_SRC" -d "$WALLET_DIR"

    if [ -f "${WALLET_DIR}/sqlnet.ora" ]; then
        sed -i "s|DIRECTORY=\"?/network/admin\"|DIRECTORY=\"${WALLET_DIR}\"|g" "${WALLET_DIR}/sqlnet.ora"
    fi

    echo "Wallet files:"
    ls -la "$WALLET_DIR"
else
    echo "Warning: wallet.zip was not found."
fi

echo "Installing Docker and starting Langfuse..."
chmod +x ./langfuse/install_docker.sh
./langfuse/install_docker.sh
systemctl start docker

docker network inspect aipoc-network >/dev/null 2>&1 || docker network create aipoc-network

EXTERNAL_IP=$(curl -s -m 10 http://whatismyip.akamai.com/ || echo "")
echo "External IP: ${EXTERNAL_IP}"

if [ -n "$EXTERNAL_IP" ]; then
    sed -i "s|localhost:3000|${EXTERNAL_IP}:3000|g" ./langfuse/docker-compose.yml
    set_env_value "LANGFUSE_HOST" "http://${EXTERNAL_IP}:3000"
else
    echo "Warning: external IP detection failed."
fi

chmod +x ./langfuse/main.sh
nohup ./langfuse/main.sh > /var/log/no1-rag-langfuse.log 2>&1 &

echo "Accepting Conda terms of service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

echo "Creating Ginza API environment..."
if ! conda_env_exists "ginza-api"; then
    conda create -n ginza-api python=3.11 -y
else
    echo "Conda environment 'ginza-api' already exists."
fi

conda activate ginza-api
pip install -r ./ginza-api/requirements.txt
chmod +x ./ginza-api/main.sh
nohup ./ginza-api/main.sh > /var/log/no1-rag-ginza-api.log 2>&1 &

echo "Creating No.1-RAG environment..."
conda activate base
if ! conda_env_exists "no.1-rag"; then
    conda create -n no.1-rag python=3.11 -y
else
    echo "Conda environment 'no.1-rag' already exists."
fi

conda activate no.1-rag
pip install -r requirements.txt
pip install gradio==5.18.0
pip install aiofiles==24.1.0
pip install defusedxml==0.7.1 pathvalidate==3.2.3 speechrecognition==3.14.1 youtube-transcript-api==1.0.1

chmod +x main.sh
nohup ./main.sh > /var/log/no1-rag.log 2>&1 &

echo "Initialization complete."

python -m nltk.downloader punkt averaged_perceptron_tagger || true
