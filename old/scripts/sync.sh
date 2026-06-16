#!/usr/bin/env bash
#
# sync.sh — rsync over SSH 양방향 동기화 (PyCharm SFTP 대체)
#
# 연결정보는 ~/.ssh/config 의 `Host rtb` alias 에만 존재 (이 스크립트엔 비밀정보 없음).
# 사전 1회: ssh-copy-id -p 5040 -i ~/.ssh/id_ed25519.pub mail-agent@3.38.195.121  (passwordless 등록)
#
# Usage:
#   scripts/sync.sh push [--dry-run] [--delete]   # 코드/노트북/docs/figures → 서버
#   scripts/sync.sh pull [--dry-run]              # 서버 → 로컬: 가벼운 산출물(figures/tables/bidding/cdf)
#   scripts/sync.sh pull-models [--dry-run]       # 서버 → 로컬: results/models (무거움, 332M)
#   scripts/sync.sh shell                         # ssh a100
#
# remote 경로가 다른 유저 소유라 push 가 권한 오류를 내면:
#   RSYNC_PATH="sudo rsync" scripts/sync.sh push   (mail-agent 에 passwordless sudo 필요)

set -euo pipefail

HOST="rtb"                    # ~/.ssh/config 의 Host alias (AWS mail-agent@3.38.195.121:5040)
REMOTE_PATH="project/rtb_ipinyou"   # 서버 홈(~) 기준 → /home/mail-agent/project/rtb_ipinyou
REMOTE="${HOST}:${REMOTE_PATH}"

# repo root = 이 스크립트의 부모 디렉터리
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PULL_LIGHT_DIRS=(
  "results/figures"
  "results/tables"
  "results/bidding"
  "results/market_price_cdf"
)

# 서버엔 rsync 가 conda 에만 있고 비대화식 PATH 에 없어 --rsync-path 로 원격 바이너리 지정.
# RSYNC_PATH 환경변수로 덮어쓸 수 있음 (예: 원격 sudo → RSYNC_PATH="sudo /home/mail-agent/conda/bin/rsync").
REMOTE_RSYNC="${RSYNC_PATH:-/home/mail-agent/conda/bin/rsync}"
rsync_base=(rsync -avz --progress --rsync-path="${REMOTE_RSYNC}")

cmd="${1:-}"
shift || true

# 나머지 인자 파싱 (--dry-run / --delete)
extra=()
for arg in "$@"; do
  case "$arg" in
    --dry-run|-n) extra+=(-n) ;;
    --delete)     extra+=(--delete) ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

case "$cmd" in
  push)
    echo ">> push  ${REPO}/  ->  ${REMOTE}/"
    "${rsync_base[@]}" ${extra[@]+"${extra[@]}"} \
      --exclude-from="${REPO}/.rsyncignore" \
      "${REPO}/" "${REMOTE}/"
    ;;

  pull)
    for d in "${PULL_LIGHT_DIRS[@]}"; do
      echo ">> pull  ${REMOTE}/${d}/  ->  ${REPO}/${d}/"
      mkdir -p "${REPO}/${d}"
      "${rsync_base[@]}" ${extra[@]+"${extra[@]}"} "${REMOTE}/${d}/" "${REPO}/${d}/"
    done
    ;;

  pull-models)
    echo ">> pull  ${REMOTE}/results/models/  ->  ${REPO}/results/models/"
    mkdir -p "${REPO}/results/models"
    "${rsync_base[@]}" ${extra[@]+"${extra[@]}"} "${REMOTE}/results/models/" "${REPO}/results/models/"
    ;;

  shell)
    exec ssh "${HOST}"
    ;;

  *)
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
