import json,re,sys,statistics as st,collections
trace,card,core,t0,t1=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5])
CORE=re.compile(r'_slice(\d+)_Core_(\d+)(?:_(.*))?$'); STEM='/model/layers.2/mlp/'
raw=json.load(open(trace))['traceEvents']
threads={(e['pid'],e['tid']):e['args']['name'] for e in raw if e['ph']=='M' and e['name']=='thread_name'}
def _isexec(e):
    m=CORE.search(threads.get((e['pid'],e['tid']),'')); return bool(m) and not m[3]
origin=min(float(e['ts']) for e in raw if e['ph']=='X' and _isexec(e))
rows=[]; gemm_stats=collections.defaultdict(list)
for e in raw:
    if e['ph']!='X': continue
    th=threads.get((e['pid'],e['tid']),''); m=CORE.search(th)
    if not m or not m[3]: continue
    a=e.get('args',{}); k=a.get('opKind','').strip()
    if k=='aicconvolutiond32' and not e['name'].startswith(('sync','barrier')):
        gemm_stats[(int(m[1]),a.get('opName','').replace(STEM,'').split('/')[0])].append((float(e['dur']),float(a.get('PCyclesPerKB',0)),int(a.get('opPCycle',0))))
    if int(m[1])!=card or int(m[2])!=core or e['name'].startswith('barrier'): continue
    ts=(float(e['ts'])-origin)/1e3
    if not (t0<=ts<=t1): continue
    d=float(a.get('opSyncDurUs',e.get('dur',0))) if e['name'].startswith('sync') else float(e.get('dur',0))
    rows.append((ts,m[3],e['name'].split(' ')[0],k,a.get('opName','').replace(STEM,'')[:34],d,a.get('opMemory',''),a.get('opOutputSize',''),a.get('PCyclesPerKB',''),a.get('opPCycle','')))
rows.sort()
print(f"# {len(rows)} events card{card} core{core} in [{t0},{t1}] ms; origin={origin}")
for r in rows:
    if r[5]<2 and r[2]=='sync': continue
    print(f"{r[0]:8.3f} {r[1]:12s} {r[2]:6s} {r[3]:22s} {r[4]:34s} dur={r[5]:8.1f}us mem={r[6]:4s} out={r[7]:>8s} cyc/KB={r[8]:>6s} pcyc={r[9]:>7s}")
print("# GEMM op stats per (card,node): n, median dur us, median cycles/KB, sum dur per core")
for key in sorted(gemm_stats):
    v=gemm_stats[key]; print(f"   card{key[0]} {key[1]:10s} n={len(v):4d} med_dur={st.median(x[0] for x in v):7.2f}us max_dur={max(x[0] for x in v):7.1f} med_cyc/KB={st.median(x[1] for x in v):6.2f} sum_dur/16cores={sum(x[0] for x in v)/16:7.0f}us")
