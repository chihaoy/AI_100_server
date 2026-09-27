#!/bin/bash
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; cd /home/wentao/workspace/AI_100_server/tools
for w in mmlu swe humaneval; do /home/chihao/qeff-venv/bin/python $S/e21/collect_routing_generic.py $S/e21/routing_$w 8 $S/e21/prompts_$w.npz > $S/e21/collect_$w.log 2>&1; echo "### $w done: $(grep -c layer2 $S/e21/collect_$w.log) prompts"; done
echo E21_COLLECT_DONE
