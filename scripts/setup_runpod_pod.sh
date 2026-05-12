#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="/workspace"
PROJECT_TGZ="${WORKSPACE}/ogs-rank-analysis-runpod.tgz"
PROJECT_DIR="${WORKSPACE}/ogs-rank-analysis"
MODELS_DIR="${WORKSPACE}/models"
KATAGO_DIR="${WORKSPACE}/katago"
KATAGO_HOME="${WORKSPACE}/katago-home"

KATAGO_VERSION="v1.16.4"
KATAGO_ZIP="katago-${KATAGO_VERSION}-cuda12.8-cudnn9.8.0-linux-x64.zip"
KATAGO_URL="https://github.com/lightvector/KataGo/releases/download/${KATAGO_VERSION}/${KATAGO_ZIP}"

MAIN_MODEL_NAME="kata1-b28c512nbt-s12704148736-d5790336910.bin.gz"
MAIN_MODEL_URL="https://media.katagotraining.org/uploaded/networks/models/kata1/${MAIN_MODEL_NAME}"

HUMAN_MODEL_NAME="b18c384nbt-humanv0.bin.gz"
HUMAN_MODEL_URL="https://github.com/lightvector/KataGo/releases/download/v1.15.0/${HUMAN_MODEL_NAME}"

JSON_GZ_NAME="sample-100k.json.gz"
JSON_GZ_URL="https://za3k.com/ogs/${JSON_GZ_NAME}"

download_if_missing() {
  local url="$1"
  local output="$2"
  if [[ -f "${output}" ]]; then
    echo "Already present: ${output}"
    return
  fi
  echo "Downloading ${url}"
  curl -fL --retry 3 --retry-delay 2 -o "${output}" "${url}"
}

echo "Workspace: ${WORKSPACE}"
mkdir -p "${WORKSPACE}" "${PROJECT_DIR}" "${MODELS_DIR}" "${KATAGO_DIR}" "${KATAGO_HOME}"

if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y curl unzip ca-certificates tmux
else
  echo "apt-get not found; assuming curl, unzip, and tmux are already available."
fi

python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"Python 3.10+ required, found {sys.version.split()[0]}")
print("Python:", sys.version.split()[0])
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "WARNING: nvidia-smi not found. This pod may not expose an NVIDIA GPU." >&2
fi

if [[ ! -f "${PROJECT_TGZ}" ]]; then
  echo "Missing ${PROJECT_TGZ}. Upload it before running this script." >&2
  exit 1
fi

echo "Unpacking project"
tar -xzf "${PROJECT_TGZ}" -C "${PROJECT_DIR}"

download_if_missing "${MAIN_MODEL_URL}" "${MODELS_DIR}/${MAIN_MODEL_NAME}"
download_if_missing "${HUMAN_MODEL_URL}" "${MODELS_DIR}/${HUMAN_MODEL_NAME}"
download_if_missing "${KATAGO_URL}" "${WORKSPACE}/${KATAGO_ZIP}"

echo "Unpacking KataGo ${KATAGO_VERSION}"
unzip -o "${WORKSPACE}/${KATAGO_ZIP}" -d "${KATAGO_DIR}"

KATAGO_BIN="${KATAGO_DIR}/katago"
if [[ ! -x "${KATAGO_BIN}" ]]; then
  FOUND_BIN="$(find "${KATAGO_DIR}" -type f -name katago -perm -111 | head -n 1 || true)"
  if [[ -n "${FOUND_BIN}" ]]; then
    KATAGO_BIN="${FOUND_BIN}"
  else
    echo "Could not find executable katago binary under ${KATAGO_DIR}" >&2
    exit 1
  fi
fi
chmod +x "${KATAGO_BIN}"

echo "KataGo binary: ${KATAGO_BIN}"
"${KATAGO_BIN}" version || true

MAIN_MODEL="${MODELS_DIR}/${MAIN_MODEL_NAME}"
HUMAN_MODEL="${MODELS_DIR}/${HUMAN_MODEL_NAME}"
CONFIG="${PROJECT_DIR}/configs/analysis_config.optimized.cfg"
JSON_GZ="${PROJECT_DIR}/data/sample-100k.json.gz"
SGF_DIR="${PROJECT_DIR}/data/sample-100k-medium-ranked-19x19-human-150moves-sgfs"
TEST_OUTPUT="${PROJECT_DIR}/results/runpod_test.csv"
FULL_OUTPUT="${PROJECT_DIR}/results/sample_100k_150moves_rank_mle_runpod.csv"

mkdir -p "${PROJECT_DIR}/data"
download_if_missing "${JSON_GZ_URL}" "${JSON_GZ}"

for path in "${MAIN_MODEL}" "${HUMAN_MODEL}" "${CONFIG}" "${JSON_GZ}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path missing after setup: ${path}" >&2
    exit 1
  fi
done

if [[ ! -d "${SGF_DIR}" ]]; then
  echo "Generating filtered SGFs from ${JSON_GZ}"
  mkdir -p "${SGF_DIR}"
  cd "${PROJECT_DIR}"
  python3 tools/jsonl_to_sgfs.py \
    "${JSON_GZ}" \
    "${SGF_DIR}" \
    --medium-ranked-19x19 \
    --human-vs-human \
    --min-moves 150
else
  echo "SGF directory already exists, skipping generation: ${SGF_DIR}"
fi

SGF_COUNT="$(find "${SGF_DIR}" -type f -name '*.sgf' | wc -l | tr -d ' ')"
if [[ "${SGF_COUNT}" -eq 0 ]]; then
  echo "No SGFs generated under ${SGF_DIR}" >&2
  exit 1
fi
echo "SGFs ready: ${SGF_COUNT}"

cat <<EOF

Setup complete.

Fixed versions:
  KataGo:      ${KATAGO_VERSION} CUDA 12.8 / cuDNN 9.8
  Main model:  ${MAIN_MODEL_NAME}
  Human model: ${HUMAN_MODEL_NAME}
  OGS sample:  ${JSON_GZ_URL}

Tiny test command:
  cd ${PROJECT_DIR}
  PYTHONPATH=tools python3 tools/analyze_rank_mle_dataset.py \\
    --katago ${KATAGO_BIN} \\
    --model ${MAIN_MODEL} \\
    --human-model ${HUMAN_MODEL} \\
    --home-data-dir ${KATAGO_HOME} \\
    --config ${CONFIG} \\
    --sgf-dir ${SGF_DIR} \\
    --output ${TEST_OUTPUT} \\
    --limit 2 \\
    --progress-every 1

Full run command:
  cd ${PROJECT_DIR}
  PYTHONPATH=tools python3 tools/analyze_rank_mle_dataset.py \\
    --katago ${KATAGO_BIN} \\
    --model ${MAIN_MODEL} \\
    --human-model ${HUMAN_MODEL} \\
    --home-data-dir ${KATAGO_HOME} \\
    --config ${CONFIG} \\
    --sgf-dir ${SGF_DIR} \\
    --output ${FULL_OUTPUT} \\
    --progress-every 1

Use tmux for the full run:
  tmux new -s rankmle
EOF
