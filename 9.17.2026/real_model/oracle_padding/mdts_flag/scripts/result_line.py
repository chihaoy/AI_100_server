import json, sys, os
p, name = sys.argv[1], sys.argv[2]
if not os.path.exists(p): print(f"=== RESULT {name}: no result.json"); sys.exit(1)
r=json.load(open(p)); t=r['timing']; l=r['logits']
print(f"=== RESULT {name}: median {t['median_ms']:.1f} ms (p10 {t['p10_ms']:.1f} p90 {t['p90_ms']:.1f}; rounds {[round(v,1) for v in t['round_medians_ms'].values()]}); logits relL2 {l['relative_l2']:.4f} argmax {l['argmax']} ref {l['reference_argmax']} next '{l['next_token']}' ref '{l['reference_next_token']}'" + (f"; bit-exact historical MXFP6 {r['bit_exact_historical_mxfp6']}" if 'bit_exact_historical_mxfp6' in r else ''))
