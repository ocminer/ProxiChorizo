ProxiChorizo 2.0.3 - native Windows x64 / AMD OpenCL

OpenCL nonce seeker for Proxima (PROX): pool mining on proxi.suprnova.cc or solo with your
own proxi node mine. Windows 10/11 x64 with an AMD Radeon or Instinct GPU and the AMD
Adrenalin (or PRO) driver, which provides OpenCL.dll. No WSL or Docker needed. The MSVC
runtime is linked statically. For NVIDIA GPUs use the windows-x86_64 (CUDA) package.

Pool (suprnova):
  1. Register at https://proxi.suprnova.cc/StartMining.html with your PROX address
     and keep the pool token. Current Blake2b jobs do not need a key file.
  2. Edit start.cmd: set TOKEN to your token.
  3. Double-click start.cmd.

PowerShell examples:
  .\start.ps1 -Token YOUR_TOKEN -Worker rig1
  .\start.ps1 -Token YOUR_TOKEN -Devices '0,1'
  .\start.ps1 -Token mine.seeker.token -Pool http://your-node:8200   # Blake2b solo
  .\proxichorizo-cl.exe --list-devices                                # the GPUs OpenCL sees

Solo: set mine.seeker.listen and mine.seeker.token in your wallet's proxi.yaml, run
`proxi node mine --workers 0`, and start ProxiChorizo with the seeker endpoint and token. Blake2b jobs need no
wallet key on the GPU machine; the wallet controller keeps that key. Details,
options and troubleshooting: USAGE.md. Stats API: add -ApiBind 127.0.0.1:4078.

Temperature: Windows does not expose AMD GPU temperatures to this miner, so the built-in
thermal limit (--max-temp) is not enforced here. Watch temperatures in AMD Software.

The hosted Windows CI runner has no AMD GPU: it checks executable startup and the pool
connection. Windows AMD GPU execution has not been tested; the same OpenCL kernels are
validated on Linux AMD GPUs. BUILD-INFO.json records the source commit and CI run.
SHA256SUMS.txt covers the files in this package; the external .sha256 file covers the ZIP.
Source and build instructions: https://github.com/ocminer/ProxiChorizo
