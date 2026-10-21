#!/usr/bin/env bash
# Publica o commit atual no Hugging Face Space.
# O Space precisa de um README com cabeçalho YAML próprio; o README do GitHub fica intacto.
#
# Uso: deploy/push_space.sh <usuario-hf>/<nome-do-space>
# Requer: token do HF com permissão de escrita em HF_TOKEN.
set -euo pipefail

SPACE="${1:?informe <usuario-hf>/<nome-do-space>}"
: "${HF_TOKEN:?defina HF_TOKEN com um token de escrita do Hugging Face}"

ROOT="$(git rev-parse --show-toplevel)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git -C "$ROOT" archive HEAD | tar -x -C "$TMP"
cp "$ROOT/deploy/space_README.md" "$TMP/README.md"

cd "$TMP"
git init -q -b main
git add -A
git commit -q -m "deploy: $(git -C "$ROOT" rev-parse --short HEAD)"
git push -q --force "https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE}" main
echo "Publicado em https://huggingface.co/spaces/${SPACE}"
