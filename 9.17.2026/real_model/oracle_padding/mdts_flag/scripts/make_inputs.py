import sys, hashlib, numpy as np; sys.path.insert(0, '/home/wentao/workspace/AI_100_server/tools')
import moe_layer0_router_oncard as R
from transformers import AutoTokenizer
out = sys.argv[1]; tok = AutoTokenizer.from_pretrained(R.HF); ids = np.asarray(R.prompts(tok, 42, 4096)[41][:128], np.int64)
ids.tofile(f'{out}/input_ids_i64.bin'); np.arange(128, dtype=np.int64).tofile(f'{out}/position_ids_i64.bin')
print('tokens', len(ids), 'sha256', hashlib.sha256(ids.tobytes()).hexdigest(), 'expected d61b64ac6a009d1f9d237baa208f524baec748bd5b3c463361d317d3c4cb7590')
