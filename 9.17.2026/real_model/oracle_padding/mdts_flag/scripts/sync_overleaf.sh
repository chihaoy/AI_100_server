#!/bin/bash
# Mirror the MLSys draft (mdts_flag/paper) into its Overleaf git clone and push, per the user's standing instruction of
# 2026-10-09. Edits made on Overleaf since the last sync stop the script, so that they are merged into mdts_flag/paper
# instead of being overwritten; .git/last_sync in the clone records the last synced commit.
# Usage: sync_overleaf.sh "<commit message>"   (OVERLEAF_DIR overrides the clone's location)
set -euo pipefail
P=$(cd "$(dirname "$0")/../paper" && pwd)
O=${OVERLEAF_DIR:-/home/wentao/workspace/AI_100_server/6ac4300ef755acfa13e67444}
MSG=${1:?usage: sync_overleaf.sh \"<commit message>\"}
export GIT_TERMINAL_PROMPT=0
cd "$O"
[ -z "$(git status --porcelain)" ] || { echo "uncommitted changes in $O: resolve them first"; exit 1; }
BASE=$(cat .git/last_sync 2>/dev/null || git rev-parse HEAD)
git pull --ff-only -q
CHANGED=$(git diff --name-only "$BASE" HEAD)
if [ -n "$CHANGED" ]; then
  echo "edited on Overleaf since the last sync ($BASE); merge these into $P first, then rerun:"
  echo "$CHANGED"; exit 2
fi
rsync -a --delete --exclude=.git "$P"/ "$O"/
git add -A
if git diff --cached --quiet; then echo "nothing to sync"; git rev-parse HEAD > .git/last_sync; exit 0; fi
git diff --cached --stat | tail -1
git commit -q -m "$MSG" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -q origin HEAD
git rev-parse HEAD > .git/last_sync
echo "pushed $(git rev-parse --short HEAD)"
