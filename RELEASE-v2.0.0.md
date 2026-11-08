# ProxiChorizo 2.0.0

**Required for Proxima library upgrade 6 (Tuesday 2026-11-10).** Update before slot 196000
(about 12:03 UTC): from then on the pool accepts only ProxiChorizo 2.0 or later.

Proxima replaces its proof of work. Until slot 196000 the work is the VRF output (as in 1.x);
from slot 197000 (about 14:54 UTC) it is the mining transaction's own ID, a BLAKE2b-256 hash
(RFC 7693) with an 8-byte nonce at its end. In between, the ledger pauses mining for about
three hours.

- **Both kinds of work in one miner.** 2.0 mines VRF jobs exactly as 1.0.8 (same kernels and
  per-card profiles) and the new hash jobs ("txid-blake2b"). The pool says which kind each job
  is; the miner switches per job, without a restart. Install it now and leave it running
  through the switch.
- **The pause.** During the ledger pause the pool sends no jobs. The miner says
  `mining paused by the ledger until slot 197000`, idles its GPUs and keeps the connection
  open; mining resumes by itself with the first hash job.
- **No key needed for hash work.** The key file is used only for VRF jobs. After the switch
  the miner runs with the pool URL, worker name and token alone; the mining OS launchers, the
  Windows script and the Docker image no longer require a key.
- **Every share is checked twice.** Each GPU passes the upgrade-6 test vectors and a CPU
  comparison of all 32 nonce positions at start (`--hash-selftest`), and every hash share is
  hashed again on the CPU before it is sent.
- **Hash rate in GH/s** on the console for hash jobs (MH/s for VRF jobs). The stats API keeps
  `hashrate` in H/s and adds `algo`, `paused` and `invalid`.
- **New options:** `--hash-selftest`, `--hash-bench [SECS]`, `--hash-launch-ms N`.
- **Solo:** hash jobs of `proxi node mine` (seeker protocol version 2) are supported as well.
- `STRATUM-SPEC.md` section 9 documents the hash jobs and shares.

## Hash rates (txid-blake2b, `--hash-bench`, 600-byte prefix)

HASHBENCH_TABLE

VRF rates are those of 1.0.8. ProxiChorizo License 1.0 is unchanged; NOTICE now names the
BLAKE2b implementation and the source of the new test vectors.
