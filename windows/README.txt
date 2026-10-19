ProxiChorizo 0.5.0 - native Windows x64 / NVIDIA CUDA

CUDA nonce seeker for Proxima (PROX): pool mining on proxi.suprnova.cc or solo with your
own proxi node mine. Windows 10/11 x64, NVIDIA Turing or newer GPU (sm_75/80/86/89/120),
driver 550 or newer. No WSL or Docker needed. If Windows reports a missing VCRUNTIME or
MSVCP DLL, install the Microsoft Visual C++ 2015-2022 x64 runtime.

Pool (suprnova):
  1. Register at https://proxi.suprnova.cc/StartMining.html with your PROX address; download
     the key file the pool gives you as proxima.key into this directory, note the token.
  2. Edit start.cmd: set TOKEN to your token (the key file is read from proxima.key).
  3. Double-click start.cmd.

PowerShell examples:
  .\start.ps1 -Token YOUR_TOKEN -Worker rig1
  .\start.ps1 -Token YOUR_TOKEN -KeyFile C:\keys\proxima.key -Devices '0,1'
  .\start.ps1 -Token mine.seeker.token -Pool http://your-node:8200 -KeyFile .\proxima.key   # solo

Solo: set mine.seeker.listen and mine.seeker.token in your wallet's proxi.yaml, run
`proxi node mine --workers 0`, and start ProxiChorizo with the wallet's key file. Details,
options and troubleshooting: USAGE.md. Stats API: add -ApiBind 127.0.0.1:4078.

The hosted Windows CI runner has no NVIDIA GPU, so Windows GPU mining is validated by the
packaged executable starting and by the Linux release packages mining live.
BUILD-INFO.json records the source commit and CI run. SHA256SUMS.txt covers the files in
this package; the external .sha256 file covers the complete ZIP.
Source and build instructions: https://github.com/ocminer/ProxiChorizo
