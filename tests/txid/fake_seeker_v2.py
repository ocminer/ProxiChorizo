#!/usr/bin/env python3
"""Loopback job server of `proxi node mine`, seeker protocol version 2 (transaction-hash work):
GET /seeker/job long-poll, POST /seeker/result (recomputed with hashlib; work key = hash[24..31] ||
hash[6..23], smaller beats), POST /seeker/report. A new job every --job-seconds.
  fake_seeker_v2.py --port 18555 --k 30 --seconds 60 --out result.json"""
import argparse, hashlib, json, os, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument('--port', type=int, default=18555); ap.add_argument('--k', type=int, default=30)
ap.add_argument('--seconds', type=float, default=60); ap.add_argument('--job-seconds', type=float, default=8)
ap.add_argument('--out', default='')
a = ap.parse_args()
lock = threading.Lock(); cond = threading.Condition(lock)
st = {'job': None, 'jobs': {}, 'accepted': 0, 'beaten': 0, 'invalid': 0, 'stale': 0, 'reports': 0, 'attempts': 0, 'reasons': [], 'nonces': set()}

def tz(b):
    w = int.from_bytes(b, 'big')
    return 64 if w == 0 else (w & -w).bit_length() - 1

def new_job(i):
    msg = os.urandom(600 + (i % 3) * 3)
    jid = int(time.time() * 1000)
    j = {'id': jid, 'version': 2, 'message': msg.hex(), 'nonce_offset': len(msg), 'k': a.k, 'beat': '', 'ttl_ms': int(a.job_seconds * 1500), 'pred': '00' * 33, 'slot': 197000 + i}
    with cond:
        st['jobs'][jid] = {'msg': msg, 'beat': None}; st['job'] = j; cond.notify_all()

class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def reply(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        after = int(self.path.split('after=')[1].split('&')[0]) if 'after=' in self.path else 0
        with cond:
            cond.wait_for(lambda: st['job'] is not None and st['job']['id'] != after, timeout=5)
            j = dict(st['job']) if st['job'] else {'id': 0}
            if j.get('id') in st['jobs']:
                bt = st['jobs'][j['id']]['beat']; j['beat'] = bt.hex() if bt else ''
        self.reply(200, j)
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path.startswith('/seeker/report'):
            with lock: st['reports'] += 1; st['attempts'] += body.get('attempts', 0)
            return self.reply(200, {})
        with lock:
            j = st['jobs'].get(body['id'])
            if j is None or st['job']['id'] != body['id']:
                st['stale'] += 1; return self.reply(409, {'error': 'stale job'})
            nonce = bytes.fromhex(body['nonce']); h = hashlib.blake2b(j['msg'] + nonce, digest_size=32).digest()
            key = h[24:32] + h[6:24]
            if len(nonce) != 8 or tz(h[24:32]) < a.k or body.get('beta', h.hex()) != h.hex() or (body['id'], nonce) in st['nonces']:
                st['invalid'] += 1; st['reasons'].append(body); return self.reply(422, {'error': 'bad solution'})
            st['nonces'].add((body['id'], nonce))
            if j['beat'] is None or key < j['beat']:
                j['beat'] = key; st['accepted'] += 1; return self.reply(200, {'accepted': True, 'beat': key.hex()})
            st['beaten'] += 1; return self.reply(200, {'accepted': False, 'beat': j['beat'].hex()})

srv = ThreadingHTTPServer(('127.0.0.1', a.port), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
t_end = time.time() + a.seconds; i = 0
while time.time() < t_end:
    new_job(i); i += 1; time.sleep(a.job_seconds)
time.sleep(2)
res = {k: st[k] for k in ['accepted', 'beaten', 'invalid', 'stale', 'reports', 'attempts']}; res['jobs'] = i; res['reasons'] = st['reasons'][:5]
print(json.dumps(res), flush=True)
if a.out: open(a.out, 'w').write(json.dumps(res, indent=1))
