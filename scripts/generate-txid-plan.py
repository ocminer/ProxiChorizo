#!/usr/bin/env python3
# ProxiChorizo License 1.0; see LICENSE.
"""Reproduce the exact OpenCL word-14/15 partial-evaluation plan.

The dependency mask follows txid::plan_snap and txid::rounds01. Constant
operations are replaced by per-job snapshots; all dependent operations and
all twelve Blake2b rounds are retained. The OpenCL compiler removes dead loads.
"""
from pathlib import Path
import argparse
S=[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],[14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3]]
def plan(sh,name):
 lines=['#define '+name+' {'];vd=[False]*16
 for i in range(16):lines.append(f'v[{i}]=P[P_VI+{i}];')
 for g in range(16):
  r,j=divmod(g,8);w=j%4;a=w;b=4+(w if j<4 else (w+1)%4);c=8+(w if j<4 else (w+2)%4);d=12+(w if j<4 else (w+3)%4)
  for h in range(2):
   mw=S[r][2*j+h];md=mw==14 or (sh and mw==15);nd=int(vd[a])+int(vd[b])+int(md);pre=0 if nd!=1 else 1 if vd[a] else 2 if vd[b] else 3;k=8*g+4*h
   vd[a]=vd[a] or vd[b] or md
   expr=f'P[107+{k}]' if not vd[a] else f'v[{a}]+P[235+{2*g+h}]' if pre==1 else f'v[{b}]+P[235+{2*g+h}]' if pre==2 else f'm[{mw}]+P[235+{2*g+h}]' if pre==3 else f'v[{a}]+v[{b}]+m[{mw}]'
   lines.append(f'v[{a}]={expr};')
   vd[d]=vd[d] or vd[a];lines.append(f'v[{d}]='+ (f'TX_ROR(v[{d}]^v[{a}],{16 if h else 32})' if vd[d] else f'P[107+{k+1}]')+';')
   vd[c]=vd[c] or vd[d];lines.append(f'v[{c}]='+ (f'v[{c}]+v[{d}]' if vd[c] else f'P[107+{k+2}]')+';')
   vd[b]=vd[b] or vd[c];lines.append(f'v[{b}]='+ (f'TX_ROR(v[{b}]^v[{c}],{63 if h else 24})' if vd[b] else f'P[107+{k+3}]')+';')
 return ' \\\n'.join(lines+['}'])+'\n'

ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');args=ap.parse_args()
p=Path(__file__).resolve().parents[1]/'opencl/txid_plan29.cl'
text=p.read_text();start=text.index('#define TX_PLAN29');end=text.index('\n__kernel',start)
generated=plan(True,'TX_PLAN29')
if args.check:
    assert text[start:end]==generated, 'txid_plan29.cl differs from the dependency graph'
    print('OpenCL plan matches the exact dependency graph')
else:
    p.write_text(text[:start]+generated+text[end:])
