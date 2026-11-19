#!/usr/bin/env python3
"""Build the native CUDA or OpenCL miner from source."""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ARCHS = ' '.join('-gencode arch=compute_%d,code=sm_%d' % (x, x)
                 for x in [61, 75, 80, 86, 89, 90, 120])
ARCHS += ' -gencode arch=compute_120,code=compute_120'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', required=True, choices=['cuda', 'opencl'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--archs', default=ARCHS)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--cxx-flags', default=(
        '-O2 -std=c++17 -Wall -Wno-unused-function -I/usr/include '
        '-static-libstdc++ -static-libgcc'))
    args = parser.parse_args()
    output = args.output.resolve()
    windows = os.name == 'nt'
    build = ROOT / 'build' / args.kind
    build.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    def run(command, cwd=ROOT):
        subprocess.run([str(x) for x in command], cwd=cwd, check=True)

    if args.kind == 'cuda':
        nvcc = os.environ.get('NVCC', 'nvcc')
        suffix = '.obj' if windows else '.o'
        main_obj = build / ('main' + suffix)
        blackwell_obj = build / ('blackwell' + suffix)
        common = [nvcc, '-O3', '-std=c++17']
        if windows:
            common.append('-Xcompiler=/MT')
        # Keep the measured Blackwell PTX module separate from the native
        # fallback images. PX_BW_PTX enables its per-card selection at runtime.
        run([*common, '-gencode', 'arch=compute_120,code=compute_120',
             '-c', 'txid_blackwell.cu', '-o', blackwell_obj])
        for attempt in range(3):
            try:
                run([*common, '--threads', str(args.threads if attempt < 2 else 1),
                     *shlex.split(args.archs), '-DPX_BW_PTX', '-c',
                     'proxichorizo.cu', '-o', main_obj])
                break
            except subprocess.CalledProcessError:
                if attempt == 2:
                    raise
                print('Retrying nvcc after unsuccessful compilation', flush=True)
        # Neither translation unit uses relocatable device code, so the host
        # linker can link their complete images directly with the CUDA runtime.
        cuda = Path(os.environ.get('CUDA_PATH') or
                    str(Path(shutil.which(nvcc)).resolve().parents[1]))
        if windows:
            run(['cl', '/nologo', '/MT', main_obj, blackwell_obj,
                 '/Fe:' + str(output), '/link', cuda / 'lib/x64/cudart_static.lib'])
        else:
            run([os.environ.get('CXX', 'g++'), '-o', output, main_obj, blackwell_obj,
                 '-L' + str(cuda / 'lib64'), '-lcudart_static', '-static-libstdc++',
                 '-static-libgcc', '-lpthread', '-ldl', '-lrt'])
    else:
        run([sys.executable, 'gen_host.py'], ROOT / 'opencl')
        if windows:
            sdk = Path(os.environ['OCL_DIR'])
            run(['cl', '/nologo', '/EHsc', '/std:c++17', '/O2', '/MT', '/W3',
                 '/utf-8', '/DNOMINMAX', '/D_CRT_SECURE_NO_WARNINGS',
                 '/I' + str(sdk / 'include'), 'proxichorizo_cl.cpp',
                 'windows_opencl_loader.cpp', '/Fe:' + str(output), '/link',
                 sdk / 'lib/OpenCL.lib', 'delayimp.lib', '/DELAYLOAD:OpenCL.dll'],
                ROOT / 'opencl')
        else:
            run([os.environ.get('CXX', 'g++'), *shlex.split(args.cxx_flags),
                 '-o', output, 'proxichorizo_cl.cpp',
                 *shlex.split(os.environ.get('CL_LDLIBS', '-lOpenCL')),
                 '-lpthread', '-ldl'], ROOT / 'opencl')
    run([output, '--help'])
    print('Build complete:', output.name)


if __name__ == '__main__':
    main()
