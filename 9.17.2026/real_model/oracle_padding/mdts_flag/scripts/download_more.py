"""Fetch small samples of four more workloads for the calibration study (streaming, first rows only).
Writes <out>/<name>.jsonl with one {"text": ...} per item. Usage: download_more.py <out dir>"""
import sys, json, itertools
from datasets import load_dataset
out = sys.argv[1]
def save(name, rows):
    with open(f'{out}/{name}.jsonl', 'w') as f:
        for r in rows: f.write(json.dumps({'text': r}, ensure_ascii=False) + '\n')
    print(name, len(rows), 'items; first:', rows[0][:90].replace('\n', ' '), flush=True)
def take(ds, n): return list(itertools.islice(ds, n))
tries = {
 'alpaca':  [lambda: [ (r['instruction'] + ('\n\n' + r['input'] if r['input'] else '')) for r in take(load_dataset('tatsu-lab/alpaca', split='train', streaming=True), 400)]],
 'news':    [lambda: ['Summarize the following article.\n\n' + r['article'] for r in take(load_dataset('abisee/cnn_dailymail', '3.0.0', split='test', streaming=True), 120)],
             lambda: ['Summarize the following article.\n\n' + r['article'] for r in take(load_dataset('cnn_dailymail', '3.0.0', split='test', streaming=True), 120)]],
 'chinese': [lambda: [f"{r['Question']}\nA. {r['A']}\nB. {r['B']}\nC. {r['C']}\nD. {r['D']}\n答案:" for s in ('chinese_history', 'chinese_literature', 'high_school_mathematics', 'college_medicine', 'marketing') for r in take(load_dataset('haonan-li/cmmlu', s, split='test', streaming=True), 60)],
             lambda: [r['premise'] + '\n' + r['hypothesis'] for r in take(load_dataset('xnli', 'zh', split='test', streaming=True), 600)]],
 'code_js': [lambda: ['Complete the following JavaScript function.\n\n' + r['prompt'] for r in take(load_dataset('bigcode/humanevalpack', 'js', split='test', streaming=True), 164)],
             lambda: ['Complete the following C++ function.\n\n' + r['prompt'] for r in take(load_dataset('bigcode/humanevalpack', 'cpp', split='test', streaming=True), 164)]],
}
for name, fns in tries.items():
    for k, fn in enumerate(fns):
        try: save(name, fn()); break
        except Exception as ex: print(name, 'attempt', k, 'failed:', type(ex).__name__, str(ex)[:160], flush=True)
