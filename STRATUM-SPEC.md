# ProxiChorizo pool Stratum protocol — specification

Version 1 (2026-11-05). This describes the wire protocol that ProxiChorizo ≥ 1.0.0 speaks
with a Proxima (PROX) mining pool, so that other miners, proxies and tools can interoperate.
It describes only what a client sees on the connection.

The key words MUST, SHOULD and MAY are used as in RFC 2119.

## 1. Overview

* Stratum v1 style: **newline-delimited JSON-RPC** over one persistent TCP connection,
  with the same message shapes as Bitcoin-style stratum pools (`mining.subscribe`,
  `mining.authorize`, `mining.set_difficulty`, `mining.notify`, `mining.submit`).
* Endpoints:
  * `stratum+tcp://proxi.suprnova.cc:3355` (plain TCP)
  * `stratum+ssl://proxi.suprnova.cc:3356` (TLS; the certificate is publicly trusted for the
    host name, so clients SHOULD verify it).

  The protocol is identical on both.
* The work is a **VRF nonce search**. For every nonce, the miner computes the
  ECVRF-EDWARDS25519-SHA512-TAI output (RFC 9381) of a message under the account's mining key.
  A share is a nonce whose 64-byte output ends in at least *k* zero bits.
* The pool issues each account a **mining key** together with an access **token**. Jobs
  are built for that key, and the pool recomputes every submitted share with it, so the miner
  must hold exactly that key (see §3).

## 2. Transport and framing

* Each message is one JSON object encoded in UTF-8 and terminated by `\n` (LF). A client
  MUST accept `\r\n` and SHOULD ignore empty lines.
* A client MUST NOT send a line longer than 10 KiB; the pool closes the connection on such a
  line and on malformed JSON.
* Client requests carry a numeric `id`. The pool's answers echo that `id` and contain
  `result` and `error`. Pool notifications carry `"id": null` and a `method`.
* Errors use the stratum form `"error": [code, "message", null]` (codes in §7).
* Clients SHOULD enable TCP keep-alive. A pool sends a new job about every slot, so a client
  MAY treat several minutes without any message as a dead connection and reconnect.
  ProxiChorizo uses 3 minutes.
* Clients MUST ignore notifications with unknown methods.

## 3. Account, token and mining key

Registration on the pool website (with the PROX payout address) returns:

* **token**: an opaque string. It is the stratum password and identifies the account.
* **key file**: a JSON document holding the account's mining key:

  ```json
  {"version": 1, "key_type": 0,
   "private_key": "<64-byte Ed25519 private key, hex: 32-byte seed || 32-byte public key>",
   "public_key":  "<32-byte Ed25519 public key, hex>",
   "holder_id":   "<hex>"}
  ```

  Only the 32-byte seed (the first 64 hex characters of `private_key`) is needed to mine.
  The secret scalar *x* and the public key *Y* are derived from it as in Ed25519 (RFC 8032):
  SHA-512 of the seed, take the first 32 bytes, clamp them, and *Y = x·B*.

The mining key is **not** the payout wallet. Rewards are booked to the account and paid to
its payout address. Both the token and the key file are shown only once, at registration;
keep them private.

Every `mining.notify` carries the account's public key. A client MUST check that it equals
the public key of the key it holds, and MUST NOT mine a job for a different key, because
none of those shares would verify.

## 4. Session

```
client                                                     pool
  |  {"id":1,"method":"mining.subscribe","params":["MyMiner/1.0"]}
  |  {"id":2,"method":"mining.authorize","params":["<address>.<worker>","<token>"]}
  |                 {"id":1,"result":[[["mining.set_difficulty","<sub id>"],["mining.notify","<sub id>"]],"<extranonce1>",6],"error":null}
  |                 {"id":2,"result":true,"error":null}
  |                 {"id":null,"method":"mining.set_difficulty","params":[<2^k>]}
  |                 {"id":null,"method":"mining.notify","params":[ ...job... ]}
  |  {"id":100,"method":"mining.submit","params":["<address>.<worker>","<job id>","<extranonce2>","<beta>"]}
  |                 {"id":100,"result":true,"error":null}
  |                 ... a new mining.notify every slot, mining.set_difficulty before it when k changes ...
```

The client MAY send subscribe and authorize back to back without waiting for the first
answer. The first difficulty and job arrive once **both** have succeeded.

### 4.1 `mining.subscribe`

* **params:** `[ "<client name/version>" ]`. The user-agent string is informational.
* **result:** `[ [["mining.set_difficulty", sub], ["mining.notify", sub]], extranonce1, extranonce2_size ]`
  * `extranonce1` is a 4-character hex string (2 bytes) that is unique among the pool's open
    connections. It is fixed for the life of the connection.
  * `extranonce2_size` is **6** (bytes).
* A client MUST refuse (and reconnect later) if `extranonce1` is not 2 bytes or the size is
  not 6.

`mining.extranonce.subscribe` is answered with `false`: extranonce1 never changes on a
connection.

### 4.2 `mining.authorize`

* **params:** `[ username, token ]`
  * `username` is `"<address>.<worker>"` or just `"<worker>"`. The address is the account's
    registered PROX address: 64 hex characters, optionally prefixed with `a/`, case-insensitive.
    If it is given, it MUST be the token's account. The token alone identifies the account.
  * The worker name is shown in the pool statistics. Characters outside `A–Z a–z 0–9 _ . -`
    are dropped, it is cut to 48 characters, and an empty name becomes `default`.
* **result:** `true`. On failure: `false` with error `[24, "unauthorized worker", null]`.
* A connection mines for **one** account. It MAY authorize several worker names of that
  account, and each submit names one of them. Authorizing a second account on the same
  connection is refused (`24`, "one account per connection").

### 4.3 `mining.set_difficulty` (pool → client)

* **params:** `[ D ]`, where *D* = 2^k is the expected number of nonces per share.
  *D* is always an exact power of two, sent as a JSON number (for example `16777216` for k = 24).
* A client derives k = log2(D) and SHOULD reject values that are not a power of two.
* A new difficulty applies **from the next `mining.notify` on**, never to the job in progress.
  The pool always sends it just before that notify.
* The pool adjusts each connection's difficulty (variable difficulty) for roughly one share
  every few tens of seconds. New connections start at k = 24. A connection that stops
  finding shares is stepped down.

### 4.4 `mining.notify` (pool → client)

**params:** `[ job_id, pubkey, pred, slot, alpha_prefix, ttl_ms, clean_jobs ]`

| # | field | type | meaning |
|---|---|---|---|
| 0 | `job_id` | string, hex | job identifier (non-zero), echoed in submits |
| 1 | `pubkey` | string, 64 hex | the account's mining public key *Y* (32 bytes), see §3 |
| 2 | `pred` | string, 66 hex | the target's predecessor output ID (33 bytes) |
| 3 | `slot` | number | the target slot (uint32) |
| 4 | `alpha_prefix` | string, 74 hex | the first 37 bytes of the VRF message: `pred ‖ slot` (slot as 4 bytes big-endian) |
| 5 | `ttl_ms` | number | milliseconds until the job expires; shares after that are stale |
| 6 | `clean_jobs` | bool | always `true`: every job replaces the previous one |

* There is exactly **one current job** per connection. When a new notify arrives, the client
  MUST stop work on the old job, because shares of the old job are answered with `21`.
* Between rounds there can be gaps without a job. After its TTL a client SHOULD idle until the
  next notify instead of mining an expired job.
* `alpha_prefix` is all a miner needs; `pred` and `slot` are informational and are
  included in it.

### 4.5 `mining.submit`

**params:** `[ username, job_id, extranonce2, beta ]`

| # | field | meaning |
|---|---|---|
| 0 | `username` | exactly a name this connection authorized (§4.2) |
| 1 | `job_id` | the current job's `job_id` |
| 2 | `extranonce2` | 12 hex characters (6 bytes), the low 48 bits of the nonce, big-endian |
| 3 | `beta` | optional: the 64-byte VRF output, 128 hex characters |

* **result:** `true` when accepted, otherwise `null` with an error from §7.
* The pool computes the output itself. If `beta` is given and differs from the pool's value,
  the share is rejected (`20`). Sending `beta` is RECOMMENDED because it catches a wrong key or
  a broken kernel at once. An empty string means "not given".
* A client SHOULD submit every share it finds for the current job, but at most **256** shares
  per job and connection are booked. Beyond that the answer is error `20` ("too many shares
  in this job"), the difficulty rises with the next job, and such shares are not counted as
  invalid.
* A client SHOULD use increasing ids (ProxiChorizo starts at 100) and match answers by `id`.

## 5. Nonce and share rule

### 5.1 Nonce

The nonce is 8 bytes, big-endian:

```
nonce = extranonce1 (2 bytes) ‖ extranonce2 (6 bytes)
```

The client chooses extranonce2 freely within its 48 bits. Each nonce may be submitted
**once** per job (a repeat is error `22`). Multi-GPU clients split the extranonce2 space.
ProxiChorizo puts the GPU index in its top 8 bits and a 40-bit counter below. Because
extranonce1 is unique per connection, connections never overlap.

### 5.2 VRF message (alpha)

```
alpha = alpha_prefix (37 bytes) ‖ nonce (8 bytes)          (45 bytes)
      = pred (33) ‖ slot (4, big-endian) ‖ extranonce1 (2) ‖ extranonce2 (6)
```

### 5.3 VRF output (beta)

beta is the ECVRF-EDWARDS25519-SHA512-TAI output of RFC 9381 (suite string `0x03`) under the
account key. Only the output is needed, not the proof:

1. **Hash to curve (try-and-increment).** For ctr = 0, 1, 2, …:
   * Compute `h = SHA-512(0x03 ‖ 0x01 ‖ Y ‖ alpha ‖ ctr ‖ 0x00)[0..32]`, where ctr is one byte.
   * Decode *h* as a compressed Edwards25519 point (RFC 8032 decoding; reject non-points).
   * Multiply by the cofactor 8.
   * The first ctr whose result is a valid point other than the identity gives *H*.
2. Compute **Gamma = x·H**.
3. Compute **beta = SHA-512(0x03 ‖ 0x03 ‖ encode(8·Gamma) ‖ 0x00)** (64 bytes).

### 5.4 Share condition

Let tz(beta) be the number of trailing zero bits of beta, read as a big-endian 512-bit number:
* Each trailing `0x00` byte, starting at beta[63], counts 8.
* The first non-zero byte from the end contributes the number of its low-order zero bits.

A nonce is a **share** if `tz(beta) ≥ k`, where *k* is the difficulty in force for the job
(§4.3). A share with tz(beta) at least the network difficulty is also a block candidate. The
pool handles that itself, so the client does nothing special.

### 5.5 Worked example

This uses the first public test vector from [`vectors.json`](vectors.json) (a test key, not
an account key):

```
seed          f08317d0c6f51dc9798368bdacd8b6e0bbc5fa1474acc9ac7a645370df932e6d
Y (pubkey)    360344b30b97c6ea9215c6e14bc9c2bdaf8c4dd682391fec8502862aabc348e4
alpha_prefix  36a0a1f98df1d0c0188d4db482d41156d62f78819fc80f288bd0b8c34078da9b3bfe373dfb
              (pred 36a0…9b3b, slot 0xfe373dfb = 4265033211)
extranonce1   fd50      extranonce2  cc112df7ae68
alpha         36a0…373dfb fd50 cc112df7ae68
ctr           1
beta          6133339c9b3f4c24c26b1bdf3dea3cf525f85632911c6f72c7a65d47713d5715
              67f9686283ccdfa276b3ac04382a33333fc940b6b90462c18cf3890e18812947
tz(beta)      0   -> a share only at k = 0
```

The submit for this nonce would be
`["<address>.<worker>", "<job id>", "cc112df7ae68", "6133…2947"]`. All vectors in
`vectors.json` give seed, pk, x, alpha, ctr, h, gamma, 8·Gamma and beta for checking an
implementation step by step.

## 6. Connection behaviour

* **Reconnect:** after a closed or dead connection, reconnect with backoff (1 s doubling to
  30 s), then subscribe and authorize again. The new connection gets a new extranonce1.
  Shares of the old connection cannot be submitted on the new one.
* **Idle:** the pool closes connections that submit no share for a long time (about ten
  minutes). A healthy miner never hits this, because the difficulty steps down.
* **Abuse protection:** a connection that sends mostly invalid shares is closed, and its
  address may be refused for some minutes. Stale shares (`21`) do not count against it.
* **Several GPUs:** one connection for all GPUs of a rig is preferred (ProxiChorizo's default),
  but several connections per account are allowed.

## 7. Error codes

| code | message | meaning / client action |
|---|---|---|
| 20 | various | other errors: wrong extranonce2 size, `beta` mismatch, unknown method, share cap reached, no free extranonce1 (retry later) |
| 21 | job not found | stale: the job ended or was replaced; count as stale, wait for the next notify |
| 22 | duplicate share | this nonce was already submitted for this job |
| 23 | low difficulty share | tz(beta) < k: check the job k and the share rule |
| 24 | unauthorized worker | wrong token, address not matching the token, unknown worker name, or a second account on one connection |
| 25 | not subscribed | submit before `mining.subscribe` succeeded |

## 8. Checklist for a new miner

1. Load the key file and derive *x* and *Y* from the seed. Check that *Y* equals `public_key`.
2. Connect, then send `mining.subscribe` and `mining.authorize("<address>.<worker>", token)`.
3. Keep extranonce1. On `mining.set_difficulty`, store k for the **next** job.
4. On `mining.notify`:
   * Check `pubkey == Y`.
   * Replace the current job.
   * Search nonces `extranonce1 ‖ extranonce2` over `alpha_prefix ‖ nonce` until TTL or the next notify.
5. For every nonce with tz(beta) ≥ k, send `mining.submit(username, job_id, extranonce2, beta)`.
6. Verify the implementation against `vectors.json` and with `--check-pool`-style probes
   (subscribe only) before mining.

The reference client is `proxichorizo.cu` (functions `stratum_client`, `stratum_notify` and
`stratum_submit`). `scripts/independent-vectors.py` is an independent Python implementation
of §5.3.
