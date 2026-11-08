#!/usr/bin/env python3
"""Loopback stratum pool for the ProxiChorizo 2.0 dual-mode test (VRF jobs, the ledger pause, then
txid-blake2b hash jobs), following the upgrade-6 stratum messages. Every hash share is recomputed with
Python's hashlib (BLAKE2b-256, RFC 7693) and checked: known job, not stale after a clean job, extranonce1
of the connection, 6-byte extranonce2, no duplicate, trailing zero bits of hash[24..31] (big endian) >= the
job's k, and the optional 4th parameter equal to the hash. VRF shares are only counted (the VRF path is
unchanged from 1.0.8 and checked by its own tests).

  fake_pool.py --port 13355 --scenario dual --pk <miner pubkey hex> --out result.json
Scenarios: dual (VRF -> pause -> hash), cold (hash from the start), pause-silent (VRF, then silence).
"""
import argparse, hashlib, json, os, random, socket, threading, time

ap = argparse.ArgumentParser()
ap.add_argument('--port', type=int, default=13355)
ap.add_argument('--scenario', default='dual')
ap.add_argument('--pk', default='00' * 32)
ap.add_argument('--seed', default='', help='miner VRF seed hex: VRF shares are then recomputed (needs PyNaCl)')
ap.add_argument('--out', default='')
ap.add_argument('--hash-seconds', type=float, default=60)
ap.add_argument('--vrf-seconds', type=float, default=20)
ap.add_argument('--pause-seconds', type=float, default=20)
ap.add_argument('--kmin', type=int, default=27)
ap.add_argument('--kmax', type=int, default=31)
ap.add_argument('--slot-seconds', type=float, default=4.0)
args = ap.parse_args()

rnd = random.Random(20261110)
vrf_beta = None
if args.seed:
    # RFC 9381 ECVRF-EDWARDS25519-SHA512-TAI output, as the 1.x mock pool computes it (independent of the miner)
    from nacl.signing import SigningKey
    from nacl.bindings import crypto_scalarmult_ed25519_noclamp, crypto_core_ed25519_add
    _seed = bytes.fromhex(args.seed); _pk = bytes(SigningKey(_seed).verify_key); args.pk = _pk.hex()
    _sc = bytearray(hashlib.sha512(_seed).digest()[:32]); _sc[0] &= 248; _sc[31] &= 127; _sc[31] |= 64; _sc = bytes(_sc)
    _P = 2**255 - 19; _D = (-121665 * pow(121666, _P - 2, _P)) % _P; _I = pow(2, (_P - 1) // 4, _P)
    def _decode(s):
        z = int.from_bytes(s, 'little'); y = (z & ((1 << 255) - 1)) % _P; sign = z >> 255
        u = (y * y - 1) % _P; v = (_D * y * y + 1) % _P; x2 = u * pow(v, _P - 2, _P) % _P
        x = pow(x2, (_P + 3) // 8, _P)
        if x * x % _P != x2: x = x * _I % _P
        if x * x % _P != x2: return None
        if x % 2 != sign: x = (-x) % _P
        return x, y
    def _add(p, q):
        x, y = p; a, b = q; t = _D * x * a * y * b % _P
        return ((x * b + y * a) * pow(1 + t, _P - 2, _P) % _P, (y * b + x * a) * pow(1 - t, _P - 2, _P) % _P)
    def _encode(p): x, y = p; return (y | ((x & 1) << 255)).to_bytes(32, 'little')
    def vrf_beta(alpha):
        for ctr in range(256):
            h = _decode(hashlib.sha512(b'\x03\x01' + _pk + alpha + bytes([ctr, 0])).digest()[:32])
            if h is None: continue
            for _ in range(3): h = _add(h, h)
            if h == (0, 1): continue
            break
        g = crypto_scalarmult_ed25519_noclamp(_sc, _encode(h))
        for _ in range(3): g = crypto_core_ed25519_add(g, g)
        return hashlib.sha512(b'\x03\x03' + g + b'\x00').digest()
lock = threading.Lock()
stats = {'scenario': args.scenario, 'vrf_jobs': 0, 'vrf_shares': 0, 'hash_jobs': 0, 'accepted': 0, 'capped': 0,
         'stale': 0, 'invalid': 0, 'duplicate': 0, 'invalid_reasons': [], 'best_work': 0, 'per_job': {},
         'prefix_lens': [], 'k_values': [], 'events': []}
jobs = {}            # job id -> {'prefix', 'k', 'en1', 'clean_gen'}
seen = set()
state = {'gen': 0, 'diff_k': args.kmin, 'cur_job': None}
clients = []

def ev(msg):
    t = time.strftime('%H:%M:%S')
    print('[pool %s] %s' % (t, msg), flush=True)
    stats['events'].append('%s %s' % (t, msg))

def send(c, obj):
    try:
        c['sock'].sendall((json.dumps(obj) + '\n').encode())
    except OSError:
        pass

def broadcast(obj):
    for c in list(clients):
        if c.get('authorized'):
            send(c, obj)

def tz64(w):
    if w == 0:
        return 64
    n = 0
    while not w & 1:
        w >>= 1; n += 1
    return n

def handle(c):
    buf = b''
    s = c['sock']
    while True:
        try:
            d = s.recv(65536)
        except OSError:
            break
        if not d:
            break
        buf += d
        while b'\n' in buf:
            line, buf = buf.split(b'\n', 1)
            if not line.strip():
                continue
            m = json.loads(line)
            meth, mid, p = m.get('method'), m.get('id'), m.get('params', [])
            if meth == 'mining.subscribe':
                c['ua'] = p[0] if p else ''
                ev('subscribe from %s, en1 %s' % (c['ua'], c['en1']))
                send(c, {'id': mid, 'result': [[['mining.set_difficulty', 's1'], ['mining.notify', 's2']], c['en1'], 6], 'error': None})
            elif meth == 'mining.authorize':
                c['user'] = p[0]
                send(c, {'id': mid, 'result': True, 'error': None})
                c['authorized'] = True
                send(c, {'id': None, 'method': 'mining.set_difficulty', 'params': [2 ** state['diff_k']]})
                if state.get('paused'):
                    send(c, {'id': None, 'method': 'client.show_message', 'params': ['mining paused by the ledger until slot 197000 (~14:54 UTC)']})
                elif state['cur_job'] is not None:
                    send(c, state['cur_job'])
            elif meth == 'mining.submit':
                with lock:
                    reply = check_submit(c, p)
                send(c, {'id': mid, 'result': reply is None, 'error': reply})
    c['dead'] = True

def check_submit(c, p):
    if len(p) < 3:
        stats['invalid'] += 1; stats['invalid_reasons'].append('short submit'); return [20, 'malformed', None]
    jid, en2 = p[1], p[2]
    j = jobs.get(jid)
    if j is None:
        stats['invalid'] += 1; stats['invalid_reasons'].append('unknown job %s' % jid); return [20, 'unknown job', None]
    if j['type'] == 'vrf':
        stats['vrf_shares'] += 1
        if vrf_beta is not None:
            beta = vrf_beta(j['alpha'] + bytes.fromhex(c['en1']) + bytes.fromhex(en2))
            work = tz64(int.from_bytes(beta[56:64], 'big')) if beta[56:64] != bytes(8) else 64
            if len(p) < 4 or p[3] != beta.hex() or work < j['k']:
                stats['invalid'] += 1; stats['invalid_reasons'].append('VRF share wrong job %s en2 %s' % (jid, en2)); return [20, 'bad VRF share', None]
            stats['vrf_verified'] = stats.get('vrf_verified', 0) + 1
        return None
    if j['gen'] < state['gen']:
        stats['stale'] += 1; return [21, 'stale job', None]
    if len(en2) != 12:
        stats['invalid'] += 1; stats['invalid_reasons'].append('en2 size'); return [20, 'extranonce2 size', None]
    key = (jid, c['en1'], en2)
    if key in seen:
        stats['duplicate'] += 1; stats['invalid'] += 1; stats['invalid_reasons'].append('duplicate %s/%s' % key[::2]); return [22, 'duplicate share', None]
    seen.add(key)
    msg = j['prefix'] + bytes.fromhex(c['en1']) + bytes.fromhex(en2)
    h = hashlib.blake2b(msg, digest_size=32).digest()
    work = tz64(int.from_bytes(h[24:32], 'big'))
    if len(p) > 3 and p[3] and p[3] != h.hex():
        stats['invalid'] += 1; stats['invalid_reasons'].append('hash param differs job %s en2 %s' % (jid, en2)); return [20, 'hash mismatch', None]
    if work < j['k']:
        stats['invalid'] += 1; stats['invalid_reasons'].append('low work %d < %d job %s en2 %s' % (work, j['k'], jid, en2)); return [23, 'low difficulty share', None]
    pj = stats['per_job'].setdefault(jid, {'k': j['k'], 'plen': len(j['prefix']), 'shares': 0, 'best': 0})
    pj['shares'] += 1; pj['best'] = max(pj['best'], work)
    stats['best_work'] = max(stats['best_work'], work)
    if pj['shares'] > 256:
        stats['capped'] += 1
    else:
        stats['accepted'] += 1
    return None

def accept_loop(srv):
    n = 0
    while True:
        s, _ = srv.accept()
        n += 1
        c = {'sock': s, 'en1': '%04x' % (0xa100 + n), 'authorized': False}
        clients.append(c)
        threading.Thread(target=handle, args=(c,), daemon=True).start()

def vrf_job(slot):
    jid = '%x' % (0x1000 + slot)
    alpha = os.urandom(37).hex()
    jobs[jid] = {'type': 'vrf', 'gen': state['gen'], 'alpha': bytes.fromhex(alpha), 'k': state['diff_k']}
    msg = {'id': None, 'method': 'mining.notify', 'params': [jid, args.pk, 'pred', slot, alpha, 10000, True]}
    state['cur_job'] = msg; stats['vrf_jobs'] += 1
    broadcast(msg)

PLENS = [600, 600, 603, 607, 633, 600, 640, 512, 520, 600]
def hash_job(slot, i, clean=True, resend=False):
    if clean:
        state['gen'] += 1
    if i % 5 == 2:  # difficulty change: applies from this job
        state['diff_k'] = rnd.randint(args.kmin, args.kmax)
        broadcast({'id': None, 'method': 'mining.set_difficulty', 'params': [2 ** state['diff_k']]})
    plen = PLENS[i % len(PLENS)]
    jid = '%x' % (0x700000 + slot * 16 + (i & 15))
    prefix = os.urandom(plen)
    jobs[jid] = {'type': 'hash', 'prefix': prefix, 'k': state['diff_k'], 'gen': state['gen']}
    msg = {'id': None, 'method': 'mining.notify', 'params': [jid, 'txid-blake2b', prefix.hex(), plen, slot, int(args.slot_seconds * 1000 * 1.5), clean]}
    state['cur_job'] = msg; stats['hash_jobs'] += 1; stats['prefix_lens'].append(plen); stats['k_values'].append(state['diff_k'])
    broadcast(msg)
    if resend:  # the same job again (the miner must not restart its extranonce2 search)
        time.sleep(0.5); broadcast(msg)
    return jid

def main():
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', args.port)); srv.listen(8)
    threading.Thread(target=accept_loop, args=(srv,), daemon=True).start()
    ev('listening on 127.0.0.1:%d, scenario %s' % (args.port, args.scenario))
    while not any(c.get('authorized') for c in clients):
        time.sleep(0.2)
    slot = 195990
    if args.scenario in ('dual', 'pause-silent'):
        t_end = time.time() + args.vrf_seconds
        while time.time() < t_end:
            vrf_job(slot); slot += 1; time.sleep(args.slot_seconds)
        state['cur_job'] = None
        if args.scenario == 'dual':
            state['paused'] = True
            ev('pause: show_message, no jobs for %.0f s' % args.pause_seconds)
            broadcast({'id': None, 'method': 'client.show_message', 'params': ['mining paused by the ledger until slot 197000 (~14:54 UTC)']})
        else:
            ev('pause without any message for %.0f s' % args.pause_seconds)
        time.sleep(args.pause_seconds)
        state['paused'] = False
    slot = 197000
    ev('hash jobs start')
    t_end = time.time() + args.hash_seconds; i = 0
    while time.time() < t_end:
        clean = (i % 4 != 3)          # every 4th job is not clean: earlier shares stay valid
        hash_job(slot, i, clean=clean, resend=(i % 6 == 1)); slot += 1; i += 1
        time.sleep(args.slot_seconds)
    time.sleep(3)
    with lock:
        stats['clients'] = [{'ua': c.get('ua'), 'en1': c['en1'], 'user': c.get('user')} for c in clients]
        stats['per_job_count'] = len(stats['per_job'])
        out = json.dumps(stats, indent=1)
    ev('done: hash jobs %d, accepted %d, capped %d, stale %d, invalid %d, vrf shares %d' % (stats['hash_jobs'], stats['accepted'], stats['capped'], stats['stale'], stats['invalid'], stats['vrf_shares']))
    if args.out:
        open(args.out, 'w').write(out)

main()
