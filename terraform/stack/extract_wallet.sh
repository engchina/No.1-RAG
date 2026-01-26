#!/bin/bash
# Terraform external data source用ウォレット抽出スクリプト
# 出力: 厳密にJSON形式のみ（stdout）

set -e

# 作業ディレクトリ（スクリプトと同じディレクトリ）
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORK_DIR"

# 一時ディレクトリを作成
rm -rf wallet_extracted >/dev/null 2>&1 || true
mkdir -p wallet_extracted

# ZIPを展開
unzip -q -o wallet_full.zip -d wallet_extracted >/dev/null 2>&1

# 不要ファイルを削除（README、Java関連ファイル）
rm -f wallet_extracted/README >/dev/null 2>&1 || true
rm -f wallet_extracted/keystore.jks >/dev/null 2>&1 || true
rm -f wallet_extracted/truststore.jks >/dev/null 2>&1 || true
rm -f wallet_extracted/ojdbc.properties >/dev/null 2>&1 || true
rm -f wallet_extracted/ewallet.pem >/dev/null 2>&1 || true

# 小さいZIPを作成
cd wallet_extracted
zip -q ../wallet_small.zip * >/dev/null 2>&1
cd ..

# 小さいZIPをbase64エンコード（改行文字を確実に除去）
WALLET_CONTENT=$(base64 -w 0 wallet_small.zip 2>/dev/null | tr -d '\r\n')

# クリーンアップ
rm -rf wallet_extracted wallet_full.zip wallet_small.zip >/dev/null 2>&1 || true

# JSONのみを出力（これがstdoutに出力される唯一の行）
printf '{"wallet_content":"%s"}\n' "$WALLET_CONTENT"
