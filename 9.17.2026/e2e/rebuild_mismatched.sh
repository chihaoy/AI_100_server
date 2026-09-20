#!/bin/bash
# rebuild any QPC whose info.json stage_widths differ from plan.json (e.g. cold QPCs built before the margin change)
E=/home/chihao/mllm/9.17.2026/e2e
for L in $(seq 0 47); do for G in hot cold; do
  if [ -f $E/L$L/$G/info.json ]; then
    W=$(python3 -c "import json;print(json.load(open('$E/L$L/$G/info.json'))['stage_widths'])")
    P=$(python3 -c "import json;p=json.load(open('$E/plan.json'))['$L'];print(p['${G}_widths'])")
    if [ "$W" != "$P" ]; then echo "L$L $G widths $W -> $P: rebuilding"; rm -rf $E/L$L/$G; fi
  fi
done; done
seq 0 47 | xargs -P 4 -I{} $E/build_layer.sh {}
