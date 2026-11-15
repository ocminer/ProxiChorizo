#!/usr/bin/env python3
"""Default production builder for native CUDA and OpenCL packages."""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
ARCHS=' '.join('-gencode arch=compute_%d,code=sm_%d'%(x,x) for x in [75,80,86,89,90,120])+' -gencode arch=compute_120,code=compute_120'

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',required=True,choices=['cuda','opencl']);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--archs',default=ARCHS);p.add_argument('--threads',type=int,default=2);p.add_argument('--require-key',action='store_true')
    p.add_argument('--cxx-flags',default='-O2 -std=c++17 -Wall -Wno-unused-function -I/usr/include -static-libstdc++ -static-libgcc')
    a=p.parse_args();output=a.output.resolve();windows=os.name=='nt';env=dict(os.environ)
    if not env.get('PX_KERNEL_BUILD_KEY'):
        assert not a.require_key, 'release build key is not configured'
        env['PX_KERNEL_BUILD_KEY']=os.urandom(32).hex()
    build=ROOT/'build'/('protected-'+a.kind);build.mkdir(parents=True,exist_ok=True);output.parent.mkdir(parents=True,exist_ok=True)
    def run(args,cwd=ROOT):subprocess.run([str(x)for x in args],cwd=cwd,env=env,check=True)
    run([sys.executable,ROOT/'scripts/protect-kernels.py','key-header',build/'kernel_key.h'])
    suffix='.obj' if windows else '.o';runtime=build/('runtime'+suffix);mono=build/('monocypher'+suffix)
    defines=['PX_KERNEL_PROTECTION']+(['PX_CUDA_PROTECTION']if a.kind=='cuda' else [])
    if windows:
        run(['cl','/nologo','/TC','/O2','/MT','/c',ROOT/'third_party/monocypher/monocypher.c','/Fo:'+str(mono)])
        run(['cl','/nologo','/EHsc','/std:c++17','/O2','/MT',*['/D'+x for x in defines],'/I'+str(build),'/c',ROOT/'kernel_protection.cpp','/Fo:'+str(runtime)])
    else:
        run(['gcc','-O2','-c',ROOT/'third_party/monocypher/monocypher.c','-o',mono])
        run([os.environ.get('CXX','g++'),'-O2','-std=c++17',*['-D'+x for x in defines],'-I'+str(build),'-c',ROOT/'kernel_protection.cpp','-o',runtime])
    if a.kind=='cuda':
        nvcc=os.environ.get('NVCC','nvcc');main_obj=build/('main'+suffix);bw=build/('blackwell'+suffix)
        prefix='/D' if windows else '-D'
        host=['-Xcompiler='+prefix+'__cudaRegisterFatBinary=px_register_fatbin','-Xcompiler='+prefix+'__cudaUnregisterFatBinary=px_unregister_fatbin']
        if windows:host+=['-Xcompiler=/MT']
        common=[nvcc,'-O3','-std=c++17',*host]
        run([*common,'-gencode','arch=compute_120,code=compute_120','-c','txid_blackwell.cu','-o',bw])
        for attempt in range(3):
            try:
                run([*common,'--threads',str(a.threads if attempt<2 else 1),*shlex.split(a.archs),'-DPX_BW_PTX','-c','proxichorizo.cu','-o',main_obj]);break
            except subprocess.CalledProcessError:
                if attempt==2:raise
                print('Retrying nvcc after unsuccessful compilation',flush=True)
        # These translation units use no relocatable device code. Host-link
        # their complete images directly; nvcc's device-link would otherwise
        # add an unused empty image outside the ordinary registration path.
        cuda=Path(os.environ.get('CUDA_PATH') or str(Path(shutil.which(nvcc)).resolve().parents[1]))
        if windows:
            run(['cl','/nologo','/MT',main_obj,bw,runtime,mono,'/Fe:'+str(output),'/link',cuda/'lib/x64/cudart_static.lib'])
        else:
            run([os.environ.get('CXX','g++'),'-o',output,main_obj,bw,runtime,mono,'-L'+str(cuda/'lib64'),'-lcudart_static','-static-libstdc++','-static-libgcc','-lpthread','-ldl','-lrt'])
    else:
        run([sys.executable,'gen_host.py'],ROOT/'opencl')
        if windows:
            sdk=Path(os.environ['OCL_DIR'])
            run(['cl','/nologo','/EHsc','/std:c++17','/O2','/MT','/W3','/utf-8','/DNOMINMAX','/D_CRT_SECURE_NO_WARNINGS','/DPX_KERNEL_PROTECTION','/I'+str(sdk/'include'),'proxichorizo_cl.cpp','windows_opencl_loader.cpp',runtime,mono,'/Fe:'+str(output),'/link',sdk/'lib/OpenCL.lib','delayimp.lib','/DELAYLOAD:OpenCL.dll'],ROOT/'opencl')
        else:
            run([os.environ.get('CXX','g++'),*shlex.split(a.cxx_flags),'-DPX_KERNEL_PROTECTION','-o',output,'proxichorizo_cl.cpp',runtime,mono,*shlex.split(os.environ.get('CL_LDLIBS','-lOpenCL')),'-lpthread','-ldl'],ROOT/'opencl')
    run([sys.executable,ROOT/'scripts/protect-kernels.py','seal',output,'--kind',a.kind,'--receipt',build/'payloads.protection.json'])
    run([sys.executable,ROOT/'tests/kernel_protection.py',output,build/'payloads.protection.json'])
    run([output,'--help'])
    run([sys.executable,ROOT/'scripts/protect-kernels.py','verify',output,'--kind',a.kind])
    (build/'kernel_key.h').unlink()
    print('Protected production build complete:',output.name)
if __name__=='__main__':main()
