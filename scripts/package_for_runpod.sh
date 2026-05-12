#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/package_for_runpod.sh [--out-dir DIR]

Creates a tarball for a RunPod KataGo/rank-MLE run:
  - ogs-rank-analysis-runpod.tgz

Defaults:
  --out-dir    /tmp
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUT_DIR="/tmp"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "${OUT_DIR}"
PROJECT_TGZ="${OUT_DIR}/ogs-rank-analysis-runpod.tgz"

cd "${REPO_ROOT}"

echo "Packaging project to ${PROJECT_TGZ}"
tar \
  --exclude='.DS_Store' \
  --exclude='__pycache__' \
  --exclude='.rank_mle_cache' \
  --exclude='*.pyc' \
  -czf "${PROJECT_TGZ}" \
  configs \
  tools \
  scripts

echo
echo "Created:"
du -h "${PROJECT_TGZ}"
echo
echo "Upload example:"
echo "  scp -P <PORT> ${PROJECT_TGZ} root@<RUNPOD_HOST>:/workspace/"
echo "  scp -P <PORT> ${REPO_ROOT}/scripts/setup_runpod_pod.sh root@<RUNPOD_HOST>:/workspace/"
