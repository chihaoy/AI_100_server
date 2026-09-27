import json, os, sys, time
from datasets import load_dataset
out = sys.argv[1]; t0 = time.time()
m = load_dataset('cais/mmlu', 'all', split='test'); print('mmlu rows', len(m), 'cols', m.column_names, f'{time.time()-t0:.0f}s', flush=True)
with open(f'{out}/mmlu_test.jsonl', 'w') as f:
    for r in m: f.write(json.dumps(dict(question=r['question'], choices=r['choices'], answer=r['answer'], subject=r['subject'])) + '\n')
s = load_dataset('princeton-nlp/SWE-bench_Lite', split='test'); print('swe-bench lite rows', len(s), 'cols', s.column_names, f'{time.time()-t0:.0f}s', flush=True)
with open(f'{out}/swebench_lite_test.jsonl', 'w') as f:
    for r in s: f.write(json.dumps(dict(instance_id=r['instance_id'], repo=r['repo'], problem_statement=r['problem_statement'], hints_text=r.get('hints_text', ''))) + '\n')
print('done', f'{time.time()-t0:.0f}s')
