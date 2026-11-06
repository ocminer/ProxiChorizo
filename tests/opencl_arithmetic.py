# ProxiChorizo License 1.0; see LICENSE.
from pathlib import Path
import ctypes,json,random,subprocess,tempfile
src=Path(__file__).resolve().parent.parent;temporary=tempfile.TemporaryDirectory(prefix='proxichorizo-arithmetic-');b=Path(temporary.name)
(b/'oracle.c').write_text('#include "shim.h"\n#include "proxichorizo.cl"\n#include "merged_scalar.h"\nvoid mul_bytes(const unsigned*x,const unsigned*y,unsigned char*out){fe a,b,h;for(int i=0;i<9;i++){a.v[i]=x[i];b.v[i]=y[i];}fe_mul(&h,&a,&b);fe_tobytes(out,&h);}\nvoid scalar(const uint8_t*x,uint8_t*y){v_xonly_scalar(x,y);}\n')
flags=['-DPX_FE=29','-DPX_FE29_KARA=1','-DPX_WG=256','-DPX_WINDOW=5','-DPX_COMPACT=1','-I'+str(src/'opencl'),'-I'+str(src/'opencl/hostref')]
subprocess.run(['gcc','-O2','-shared','-fPIC',*flags,str(b/'oracle.c'),'-o',str(b/'oracle.so')],check=True)
a=ctypes.CDLL(str(b/'oracle.so'));a.scalar.argtypes=[ctypes.c_void_p,ctypes.c_void_p];a.mul_bytes.argtypes=[ctypes.c_void_p]*3;out=ctypes.create_string_buffer(32);L=2**252+27742317777372353535851937790883648493;P=2**255-19;r=random.Random(107)
cases=[0,1,2,2**256-1]+[k*L+d for k in range(16) for d in [-1,0,1] if 0<=k*L+d<2**256]+[r.getrandbits(256) for _ in range(100000)]
for s in cases:
 a.scalar(s.to_bytes(32,'little'),out);t=8*s%L;assert int.from_bytes(out.raw,'little')==min(t,L-t)
M=2**30-1;edges=[[0]*9,[M]*9,[(1<<29)-1]*9,[M if i%2 else 0 for i in range(9)]]+[[M if i==j else 0 for i in range(9)] for j in range(9)]
fields=[(x,y) for x in edges for y in edges]+[([r.randrange(1<<30) for _ in range(9)],[r.randrange(1<<30) for _ in range(9)]) for _ in range(20000)]
for x,y in fields:
 a.mul_bytes((ctypes.c_uint*9)(*x),(ctypes.c_uint*9)(*y),out);xi=sum(t<<(29*i) for i,t in enumerate(x));yi=sum(t<<(29*i) for i,t in enumerate(y));assert int.from_bytes(out.raw,'little')==xi*yi%P
print('PASS: %d scalar reductions and %d bounded field products against Python integers' % (len(cases),len(fields)))
temporary.cleanup()
