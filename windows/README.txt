ProxiChorizo 2.0.2 - native Windows x64 / NVIDIA CUDA

CUDA nonce seeker for Proxima (PROX): pool mining on proxi.suprnova.cc or solo with your
own proxi node mine. Windows 10/11 x64, NVIDIA Turing or newer GPU (sm_75/80/86/89/90/120),
a compatible NVIDIA driver for CUDA 12.9 and your GPU. No WSL or Docker needed.
The CUDA and MSVC runtimes are linked statically.

Pool (suprnova):
  1. Register at https://proxi.suprnova.cc/StartMining.html with your PROX address
     and keep the pool token. Current Blake2b jobs do not need a key file.
  2. Edit start.cmd: set TOKEN to your token.
  3. Double-click start.cmd.

PowerShell examples:
  .\start.ps1 -Token YOUR_TOKEN -Worker rig1
  .\start.ps1 -Token YOUR_TOKEN -Devices '0,1'
  .\start.ps1 -Token mine.seeker.token -Pool http://your-node:8200   # Blake2b solo

Solo: set mine.seeker.listen and mine.seeker.token in your wallet's proxi.yaml, run
`proxi node mine --workers 0`, and start ProxiChorizo with the seeker endpoint and token. Blake2b jobs need no
wallet key on the GPU machine; the wallet controller keeps that key. Details,
options and troubleshooting: USAGE.md. Stats API: add -ApiBind 127.0.0.1:4078.

The hosted Windows CI runner smoke-tests executable startup and has no NVIDIA GPU.
Windows GPU hardware execution has not been tested. Linux GPU validation is recorded separately.
BUILD-INFO.json records the source commit and CI run. SHA256SUMS.txt covers the files in
this package; the external .sha256 file covers the complete ZIP.
Source and build instructions: https://github.com/ocminer/ProxiChorizo
