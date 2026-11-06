# Independent field arithmetic + libsodium point fixtures.
# ProxiChorizo License 1.0; see LICENSE.
import hashlib,json,sys
from pathlib import Path
from nacl.bindings import crypto_scalarmult_ed25519_noclamp,crypto_core_ed25519_add
P=2**255-19;L=2**252+27742317777372353535851937790883648493;D=(-121665*pow(121666,P-2,P))%P;I=pow(2,(P-1)//4,P)
def add(p,q):
 x,y=p;a,b=q;t=D*x*a*y*b%P
 return ((x*b+y*a)*pow(1+t,P-2,P)%P,(y*b+x*a)*pow(1-t,P-2,P)%P)
def extended_add(p,q):
 X,Y,Z,T=p;U,V,W,S=q
 A=(Y-X)*(V-U)%P;B=(Y+X)*(V+U)%P;C=2*D*T*S%P;dd=2*Z*W%P
 E=(B-A)%P;F=(dd-C)%P;G=(dd+C)%P;H=(B+A)%P
 return E*F%P,G*H%P,F*G%P,E*H%P
def mul(p,s):
 x,y=p;b=(x,y,1,x*y%P);q=(0,1,1,0)
 while s:
  if s&1:q=extended_add(q,b)
  b=extended_add(b,b);s>>=1
 x,y,z,_=q;iz=pow(z,P-2,P);return x*iz%P,y*iz%P
def decode(s):
 z=int.from_bytes(s,'little');y=(z&((1<<255)-1))%P;sign=z>>255;v=(D*y*y+1)%P;x2=(y*y-1)*pow(v,P-2,P)%P;x=pow(x2,(P+3)//8,P)
 if x*x%P!=x2:x=x*I%P
 if x*x%P!=x2:return None
 if x%2!=sign:x=-x%P
 return x,y
def encode(p):return (p[1]|((p[0]&1)<<255)).to_bytes(32,'little')
def h(n):return n.to_bytes(32,'little').hex()
points=[];tor=None
for i in range(100):
 p=decode(hashlib.sha256(('point edge '+str(i)).encode()).digest())
 if p is None:continue
 points.append(p)
 if tor is None:
  t=mul(p,L)
  if mul(t,4)!=(0,1) and mul(t,8)==(0,1):tor=t
 if len(points)>=5 and tor is not None:break
assert tor is not None
T=[mul(tor,i) for i in range(8)]
B=decode(bytes.fromhex('5866666666666666666666666666666666666666666666666666666666666666'))
raw=T+[add(p,t) for p in [B]+points[:5] for t in T]
keys=[2**254,2**255-8,int.from_bytes(hashlib.sha512(b'OpenCL scalar edge').digest()[:32],'little')&((1<<255)-8)|1<<254,(-pow(64,-1,L))%L]
# Exercise ladder prefixes immediately beside multiples of the subgroup order.
for multiple in [1,2,3,4,6,7]:
 m=multiple*L-(multiple*L)%8
 x0=((m//8)*pow(8,-1,L))%L
 choices=[x0+k*L for k in range(8) if 2**254<=x0+k*L<2**255 and (x0+k*L)%8==0]
 keys.append(choices[0] if choices else x0)
if "--extended" in sys.argv[2:]:
 keys += [0,1,2,3,4,7,8,16,32,64,2**255,2**256-1]
 for k in [1,2,4,8]:keys += [k*L-1,k*L,k*L+1]
 keys += [int.from_bytes(hashlib.sha256(('extra scalar '+str(i)).encode()).digest(),'little') for i in range(12)]
 keys=list(dict.fromkeys(keys))
out=[]
for scalar in keys:
 for p in raw:
  expected=mul(p,64*scalar);valid=expected!=(0,1)
  # Cross-check the independent complete-group oracle with libsodium where
  # its scalar API is defined (nonzero subgroup scalar, bit255 clear).
  q=mul(p,8)
  if q!=(0,1) and scalar%L and scalar<2**255:
   cg=crypto_scalarmult_ed25519_noclamp(bytes.fromhex(h(scalar)),encode(q))
   for _ in range(3):cg=crypto_core_ed25519_add(cg,cg)
   assert decode(cg)==expected
  out.append(' '.join([h(scalar),h(p[0]),h(p[1]),h(expected[0]),h(expected[1]),str(int(valid))]))
Path(sys.argv[1]).write_text('\n'.join(out)+'\n')
print('Generated',len(out),'independent points;',len(T),'torsion elements;',len(keys),'scalars; subgroup boundaries, including Q=-P.')
