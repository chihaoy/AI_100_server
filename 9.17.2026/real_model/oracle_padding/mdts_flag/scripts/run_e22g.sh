#!/bin/bash
# E22g: sweep every collected 256- and 512-token prompt through the tight-capacity per-card dynamic QPCs (160/16, 304/48)
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python; E=$S/e22
$PY - <<EOP
import json
E="$E"
for T in (256, 512):
    spec = json.load(open(f"{E}/spec_sweep{T}.json"))
    for c in spec['cases']: c['qpc'] = f"{E}/qpc/T{T}_dyncard4t"
    json.dump(spec, open(f"{E}/spec_sweep{T}_tight.json", "w"), indent=1)
EOP
for T in 256 512; do $PY $S/e7_timing.py $E/spec_sweep${T}_tight.json $E/sweepT${T}_tight.json 1 2>&1 | grep -E "round 0|FAILED|Traceback|Error" | awk '{print $3, $5, $NF, $(NF-6)}' > $E/sweepT${T}_tight.txt
  echo "T=$T tight sweep: $(wc -l < $E/sweepT${T}_tight.txt) prompts, counts exact $(grep -c True $E/sweepT${T}_tight.txt), latency median $(awk '{print $2}' $E/sweepT${T}_tight.txt | sort -g | awk '{v[NR]=$1} END{print v[int((NR+1)/2)]}')"; done
echo E22G_DONE
