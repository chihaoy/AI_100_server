#!/usr/bin/env python3
"""档位表能否分解为「形状 x 排名」—— 决定路线②要编 N 张图还是 1 张图。

跨域直接套用档位表丢 37.5%。本实验检验那 37.5% 是"形状不对"还是"分错了人":
借用 A 域的档位形状(每层降序多重集),按 B 域自己的热度排名重新分配。
若丢弃率回到 B 域自拟合水平,则说明形状跨域可复用,只需换一张置换表。

限制:换排名 = 换 expert 权重所在槽位 = 权重重排,需重新装载,
      只能按部署/按批次切换,做不到逐请求。
"""
import numpy as np, sys
sys.path.insert(0,'/home/chihao/mllm/tools')
from moe_tier_crossdomain import windows, fit_tiers, evaluate
W, Q = 48, 95
TIERS=[0,2,4,8,16,32,W]
D={}
for b in ['gsm8k','humaneval']:
    r,E = windows('/home/chihao/mllm/perop_moe/routing/routing_%s.npz'%b, W)
    n=len(r); D[b]={'fit':r[:n//2],'eval':r[n//2:]}
L=D['gsm8k']['fit'].shape[1]

t_g = fit_tiers(D['gsm8k']['fit'],     TIERS, Q)
t_h = fit_tiers(D['humaneval']['fit'], TIERS, Q)

sg = np.sort(t_g,axis=1)[:,::-1]; sh = np.sort(t_h,axis=1)[:,::-1]
print('每层档位形状(降序向量)跨域差异: 平均 L1 距离 %.1f 行/层' % np.abs(sg-sh).sum(1).mean())
print('  GSM8K 每层总行数 %.0f   HumanEval %.0f' % (sg.sum(1).mean(), sh.sum(1).mean()))
print('  逐层形状完全相同的层数: %d/%d' % ((sg==sh).all(1).sum(), L))

need_h = np.percentile(D['humaneval']['fit'], Q, axis=0)
t_perm = np.zeros_like(t_g)
for l in range(L):
    order = np.argsort(-need_h[l])
    t_perm[l, order] = sg[l]

print()
print('%-34s%9s%9s%9s' % ('方案','丢弃率','行/层','vs当前'))
print('-'*61)
for name, t in [('HumanEval 自己的表(上限)', t_h),
                ('GSM8K 表直接用(跨域)', t_g),
                ('GSM8K 的形状 + HumanEval 的排名', t_perm)]:
    d,rows = evaluate(D['humaneval']['eval'], t)
    print('%-34s%8.2f%%%9.0f%8.2fx' % (name, d*100, rows, 128*W/rows))
