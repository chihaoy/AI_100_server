import sys; sys.path.extend(['/opt/qti-aic/dev/lib/x86_64','/opt/qti-aic/dev/python'])
import qaicrt, QAicApi_pb2 as api
q=qaicrt.Qpc(sys.argv[1]); st,data=q.getIoDescriptor(); d=api.IoDesc(); d.ParseFromString(bytes(data)); b=d.selected_set.bindings
print(f"### bindings {sys.argv[1].rstrip('/').split('/')[-1]}: {len(b)} total, {sum(x.name.startswith('past_') for x in b)} KV, others {[x.name for x in b if not x.name.startswith('past_')]}")
