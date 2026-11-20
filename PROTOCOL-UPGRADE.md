# PROTOCOL-UPGRADE: aligned-tail jobs for Proxima pools

Wire extension: **`proxima.aligned-tail.v1`**. Implemented by ProxiChorizo 2.1.0
for NVIDIA CUDA and AMD OpenCL, including Linux and Windows and their mining-OS
and container packages. This document contains public protocol details and
synthetic examples only. It is intended for other pool and miner implementers.

## What changes

The pool can offer a canonically serialized mining transaction whose final
BLAKE2b block consists of 120 zero bytes followed by the existing eight-byte
nonce. The miner precomputes the earlier blocks and can simplify work involving
the zero message words. All twelve BLAKE2b rounds remain exact.

This changes the transaction layout, not the hash function, difficulty rule,
nonce width, job lifetime, share accounting or reward rules. “Long job” describes
the longer serialized preimage, not a longer deadline. Neither 600 bytes nor any
other fixed total length is required. Ordinary jobs remain supported alongside it.

This capability does not itself activate a consensus upgrade. Construct and
validate the transaction under the rules of the network your node is running.
There is no upgrade-7 activation condition in this wire protocol.

## 1. Negotiate per connection

Before the usual subscribe and authorize requests, the client may send:

```json
{"id":3,"method":"mining.configure","params":[["proxima.aligned-tail.v1"],{}]}
```

The server explicitly accepts the capability for that connection:

```json
{"id":3,"result":{"proxima.aligned-tail.v1":true},"error":null}
```

If disabled or unsupported, return `false`, an empty result object, or an ordinary
unknown-method error. A client counts only a boolean `true` with no RPC error as
acceptance. A miner need not wait for this optional response before sending its
normal subscribe/authorize messages. If an old server closes on an unknown method
before subscribe completes, ProxiChorizo retries the original handshake.

Offer padded jobs only to connections that explicitly requested and were accepted
for the extension. Do not infer support solely from a miner name or version. A late
configure request takes effect from the next issued job; it does not change the
template underlying a job already sent. Reset negotiation state on reconnect.
ProxiChorizo requests the capability again on a fresh connection, even after a
previous refusal. The one immediate legacy-handshake retry described above omits
configure; later connections negotiate again. A pool upgrade therefore does not
require a miner restart.

ProxiChorizo 2.1.0 requests the extension by default on **all platforms**, including
Pascal and mixed-GPU rigs. It selects the fastest valid available kernel separately
on each GPU. `PX_ALIGNED_TAIL=0` is an optional diagnostic opt-out. This environment
variable is not part of the wire protocol.

## 2. Keep the existing Stratum messages

The normal dialect is `mining.subscribe`, `mining.authorize`, `mining.notify`, and
`mining.submit`, as described in [STRATUM-SPEC](STRATUM-SPEC.md#9-hash-work-txid-blake2b-proxima-library-upgrade-6).
Do not mix the separate native `login`/`job`/`submit` dialect into this connection.
This extension does not specify or require that native dialect.

An unchanged subscribe result example is:

```json
{"id":1,"result":[[],"0102",6],"error":null}
```

`extranonce1` is two bytes; `extranonce2` is six bytes. Authentication, credentials,
fees, vardiff policy and connection limits remain the pool's existing policy.

Notify still contains exactly seven fields:

```text
{"id":null,"method":"mining.notify","params":[
  "job-42","txid-blake2b","<hex of the actual prefix>",504,123456,10000,true
]}
```

The fields are job ID, algorithm, prefix hex, prefix length in **bytes**, ledger
slot, TTL in milliseconds, and clean-jobs flag. `504` is an example. The length
must equal the decoded prefix length. Do not add padding metadata, a replacement
nonce width, or extra notify fields. `mining.set_difficulty` and its next-job
application remain unchanged.

For each padded job the following invariant must hold:

```text
L = len(prefix)
L >= 120
L % 128 == 120
prefix[L - 120 : L] == 120 zero bytes
message = prefix || extranonce1[2] || extranonce2[6]
```

The complete message therefore ends on a 128-byte BLAKE2b boundary. Prefixes of
504 and 632 bytes are typical examples. Each miner must check the actual bytes
on every job; an acknowledgement alone is not evidence of the invariant.

## 3. Construct the real transaction, then derive its preimage

Use the chain's canonical transaction and inline-data serializers. The **last
transaction-level constraint** must contain an inline-data payload of:

```text
zero_bytes(pad) || nonce[8]
```

The padding belongs inside that payload. Simply appending bytes to a serialized
preimage, or inserting an unrelated zero-only constraint before an ordinary
eight-byte nonce constraint, does not establish the required invariant.

One possible bounded construction algorithm:

```text
for pad in 120 .. implementation_limit:
    template = clone(normal_unsigned_mining_transaction)
    template.last_nonce_constraint = InlineData(zero_bytes(pad) || zero_bytes(8))
    p0 = signer_bound_preimage(canonical_serialize(template), signer_identity)
    if len(p0) % 128 != 0 or p0[-128:-8] != zero_bytes(120):
        continue
    replace ONLY the final eight payload bytes with ff ff ff ff ff ff ff ff
    p1 = signer_bound_preimage(canonical_serialize(template), signer_identity)
    require len(p1) == len(p0)
    require p1[:-8] == p0[:-8] and p1[-8:] == eight_ff_bytes
    retain immutable template and pad
    return prefix = p0[:-8]
return ordinary_template_for_this_round
```

The search must account for the inline-data encoding and every outer length
prefix, which can change when a payload grows. Preserve all transaction fields,
signer identity, outputs, amounts, fee/delegation choices, parent inputs and any
other constraints. Use the running node's transaction-size and consensus limits.

The official transaction preimage and ID construction are in Proxima's
[`ledger/txbuildercore/sign.go`](https://github.com/lunfardo314/proxima/blob/v1.7.0/ledger/txbuildercore/sign.go).
The extension is independent of a particular pool codebase.

## 4. Bind each submitted share to the issued template

Store a job record for each connection containing its exact prefix and immutable
transaction template, padding choice, extranonce1, target/difficulty, and expiry.
The ordinary and padded versions of the same round may share a job ID **across
different connections**. Never resolve a submitted job ID through a single global
template map that forgets which layout that connection received.

The submit format remains:

```text
{"id":100,"method":"mining.submit","params":[
  "ADDRESS.WORKER","job-42","030405060708","<64-hex complete BLAKE2b-256 digest>"
]}
```

Reconstruct `nonce = extranonce1 || extranonce2` and recompute:

```text
hash = BLAKE2b-256(prefix || nonce)
work = trailing_zero_bits(uint64_big_endian(hash[24:32]))
accept share iff work >= k and all ordinary job/auth/duplicate/expiry checks pass
```

Use the unkeyed 32-byte BLAKE2b digest with no salt or personalization. Do not use
a 64-byte BLAKE2b digest truncated to 32 bytes. The optional claimed hash does not
replace the pool's own computation. Maintain the existing duplicate key for the
job/connection/extranonce2 and the existing share-weight calculation.

For network-difficulty solutions, put the winning nonce into the **same padded
template**, sign it, and derive its actual transaction ID. Confirm that the signed
transaction's work matches the submitted solution before sending it to the node.
Never sign the ordinary template using a nonce found for the padded template.
Perform ordinary node acceptance, inclusion, confirmation and accounting checks.

## 5. Fallback and deployment

- Keep ordinary jobs for clients without the capability. Enabling the option must
  not disconnect them or silently reinterpret their existing jobs.
- If padding construction fails for a round, send that round's ordinary job even
  to opted-in connections. The miner must select its kernel from the actual job.
- Do not retroactively change a template associated with an outstanding job.
- Expire old jobs and pending submissions using the existing clean/TTL rules.
- A rollback can stop offering the extension; use normal job transitions or
  reconnects to reset the connection's negotiated state.
- Check the larger signed-transaction size against your network's current limits
  and observe the first production confirmations and account credits. A staging
  chain that disables reward calculation cannot demonstrate production payouts.

## 6. Interoperability checks

Before rollout, verify at least:

1. Ordinary and opted-in clients mine in the same rounds; neither layout is used
   to validate the other connection's shares, even with a shared job ID.
2. Configure before subscribe, after the first job, declined, missing, malformed
   and unsupported-method replies behave as described.
3. Both 504-byte and 632-byte prefixes satisfy the invariant and produce exact
   full hashes; ordinary fallback works within an opted-in connection.
4. The two probe nonces change only the final eight preimage bytes.
5. Wrong claimed hashes, duplicates, expired jobs and below-target shares are
   rejected by the normal rules.
6. At least one GPU-found padded solution is signed, accepted by the real node,
   included and confirmed. Check the matching pool block/accounting record.
7. Test multi-GPU nonce partitioning if kernels enumerate their ranges differently.

Public synthetic hash vectors: [`vectors-aligned-tail.json`](vectors-aligned-tail.json).
Verify them without a GPU or network connection:

```sh
python3 tests/txid/aligned_tail_vectors.py
```

These vectors test bytes, hash order and the share rule. They are deliberately
synthetic hash preimages, **not valid signed transactions or block proofs**.
The miner's shared transport tests and GPU/CPU zero-tail tests are documented in
[`tests/txid/README.md`](tests/txid/README.md).
