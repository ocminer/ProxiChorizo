ProxiChorizo 2.1.1 - native Windows x64 / NVIDIA CUDA

CUDA nonce seeker for Proxima (PROX): pool mining on proxi.suprnova.cc or solo with your
own proxi node mine. Windows 10/11 x64, NVIDIA GPU with sm_61/75/80/86/89/90/120,
a compatible NVIDIA driver for CUDA 12.9 and your GPU. No WSL or Docker needed.
The CUDA and MSVC runtimes are linked statically.

Pool (suprnova):
  1. Copy your PROX address (64-hex holder ID). No registration or key file is needed.
     Current setup: https://proxi.suprnova.cc/StartMining.html . Existing tokens still work.
  2. Edit start.cmd: set TOKEN to your PROX address or existing token.
  3. Double-click start.cmd.

PowerShell examples:
  .\start.ps1 -Token YOUR_PROX_ADDRESS -Worker rig1
  .\start.ps1 -Token YOUR_PROX_ADDRESS -ExtraArgs @("--no-autotune")
  .\start.ps1 -Token YOUR_PROX_ADDRESS -Devices '0,1'
  .\start.ps1 -Token mine.seeker.token -Pool http://your-node:8200   # Blake2b solo

Startup: --no-autotune skips legacy VRF tuning trials, using a valid cache or the
platform default. The brief Blake2b comparison and mandatory verification remain.
Current Blake2b jobs need no Seed/KeyFile. Long jobs are negotiated automatically;
ordinary jobs continue to work. Do not combine --no-autotune with --autotune.

Solo: set mine.seeker.listen and mine.seeker.token in your wallet's proxi.yaml, run
`proxi node mine --workers 0`, and start ProxiChorizo with the seeker endpoint and token. Blake2b jobs need no
wallet key on the GPU machine; the wallet controller keeps that key. Details,
options and troubleshooting: USAGE.md. Stats API: add -ApiBind 127.0.0.1:4078.

The hosted Windows CI runner smoke-tests executable startup and has no NVIDIA GPU.
Windows GPU hardware execution has not been tested. Linux GPU validation is recorded separately.
BUILD-INFO.json records the source commit and CI run. SHA256SUMS.txt covers the files in
this package; the external .sha256 file covers the complete ZIP.
Source and build instructions: https://github.com/ocminer/ProxiChorizo
