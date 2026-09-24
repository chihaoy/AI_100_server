#!/usr/bin/env bash
# Grant one user AI 100 access and read/execute access to existing MoE assets.
# Run from an interactive terminal: sudo bash tools/moe_grant_access.sh wentao
# No files are made writable and no other user's home is made listable.
set -euo pipefail

if [[ $# != 1 || $EUID != 0 ]]; then
  echo "Usage: sudo bash $0 <username>" >&2
  exit 2
fi
target=$1
id "$target" >/dev/null
getent group qaic >/dev/null
command -v setfacl >/dev/null
command -v getfacl >/dev/null

parents=()
assets=()
devices=()
for path in /home/chihao /home/chihao/models /home/chihao/models/qwen3_30b_a3b \
            /home/chihao/models/bench /home/chihao/mllm /home/chihao/mllm/research \
            /home/chihao/mllm/9.17.2026 /home/chihao/mllm/research/9.17.2026 \
            /home/chihao/mllm/perop_moe /home/chihao/mllm/research/perop_moe; do
  [[ ! -d "$path" ]] || parents+=("$path")
done
for path in /home/chihao/qeff-venv /home/chihao/models/qwen3_30b_a3b/hf \
            /home/chihao/models/bench/gsm8k_test.jsonl \
            /home/chihao/mllm/9.17.2026/e2e /home/chihao/mllm/research/9.17.2026/e2e \
            /home/chihao/mllm/9.17.2026/real_l0 /home/chihao/mllm/research/9.17.2026/real_l0 \
            /home/chihao/mllm/perop_moe/routing /home/chihao/mllm/research/perop_moe/routing; do
  if [[ -e "$path" ]]; then
    assets+=("$path")
  else
    printf 'Not present (skipping): %s\n' "$path"
  fi
done
for path in /dev/accel/accel{0..3}; do
  [[ ! -c "$path" ]] || devices+=("$path")
done
if [[ ${#devices[@]} == 0 ]]; then
  echo 'No AI 100 device nodes found; nothing changed.' >&2
  exit 1
fi

# Save existing ACLs before changing any permissions. The backup is root-only.
backup=$(mktemp -d /tmp/moe-access-backup.XXXXXX)
getfacl -p -- "${parents[@]}" "${devices[@]}" > "$backup/acls.txt"
if [[ ${#assets[@]} != 0 ]]; then
  getfacl -R -p -- "${assets[@]}" >> "$backup/acls.txt"
fi
id "$target" > "$backup/original-groups.txt"
printf 'ACL backup: %s/acls.txt\n' "$backup"

# Group membership persists across device recreation and future login sessions.
usermod -a -G qaic "$target"
# Existing processes do not inherit newly added groups: these ACLs work now.
setfacl -m "u:$target:rw" -- "${devices[@]}"
if [[ ${#parents[@]} != 0 ]]; then
  setfacl -m "u:$target:--x" -- "${parents[@]}"
fi
if [[ ${#assets[@]} != 0 ]]; then
  setfacl -R -m "u:$target:r-X" -- "${assets[@]}"
fi
printf '\nAccess granted to %s. Existing sessions can use the device ACLs now.\n' "$target"
echo 'Log out and back in later to inherit persistent qaic group membership.'
echo 'Asset access is read/execute only; write new results in your own workspace.'
echo 'Symlink targets outside the listed asset directories may need separate access.'
printf 'To restore previous ACLs: sudo setfacl --restore=%q\n' "$backup/acls.txt"
echo 'The ACL restore does not remove the newly added qaic group membership.'
