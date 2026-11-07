# ProxiChorizo 1.0.8

Windows fix: full hashrate while mining.

- On Windows, mining ran at about half the speed shown during tuning, with shares
  arriving in bursts followed by pauses. Since 1.0.4 the miner sleeps while a GPU
  launch runs, to avoid busy-waiting a CPU core. On Windows the NVIDIA driver holds
  queued GPU work until the program next talks to it, so the card sat idle for the
  whole sleep. 1.0.8 hands every launch to the GPU immediately.
- On Windows the miner now requests 1 ms timer resolution, so its short waits are
  not stretched to the default ~15.6 ms.
- Job-change aborts are delivered to the GPU immediately on Windows as well.

GPU arithmetic and kernels are unchanged from 1.0.7. Linux CPU usage is unchanged.
ProxiChorizo License 1.0 and third-party notices are unchanged.
