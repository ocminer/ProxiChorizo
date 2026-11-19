// ProxiChorizo License 1.0; see LICENSE.
// Real shared CUDA/OpenCL transport against a loopback fixture, no GPU needed.
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <random>
#include <fstream>
#include <sstream>
#include <map>
#include <cmath>
#include <cerrno>
#include <ctime>
#include <condition_variable>
#include <cassert>
#include "../../txid_zero_tail.h"
#include "../../opencl/host_common.inc"

template<class F> static void await(F f) {
    auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!f()) { assert(std::chrono::steady_clock::now() < end); std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
}
static std::string line(int fd) {
    std::string s; char c;
    do { assert(recv(fd,&c,1,0)==1); if (c!='\n') s+=c; } while(c!='\n');
    return s;
}
static void sendline(int fd, const std::string &s) {
    std::string wire=s+"\n"; size_t off=0;
    while(off<wire.size()){auto n=send(fd,wire.data()+off,wire.size()-off,MSG_NOSIGNAL);assert(n>0);off+=n;}
}
static JV json(const std::string &s) { JV v; size_t off=0; assert(jparse(s,off,v)); return v; }
static void session(const std::string &ack, bool accepted, bool closes, bool late_ack = false) {
    int listener=socket(AF_INET,SOCK_STREAM,0); assert(listener>=0);
    sockaddr_in a{}; a.sin_family=AF_INET; a.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
    assert(bind(listener,(sockaddr*)&a,sizeof a)==0); assert(listen(listener,2)==0);
    socklen_t len=sizeof a; assert(getsockname(listener,(sockaddr*)&a,&len)==0);
    Shared sh; sh.have_key=false; sh.user="test.worker"; sh.token="x";
    assert(parse_url("stratum+tcp://127.0.0.1:"+std::to_string(ntohs(a.sin_port)),sh.url));
    std::thread client(stratum_client,std::ref(sh));
    auto accept_one = [&] { assert(sock_readable(listener,5000)); int fd=accept(listener,nullptr,nullptr);assert(fd>=0);sock_timeout(fd,5);return fd; };
    int fd=accept_one(); auto request=json(line(fd));
    assert(request.get("method")->s=="mining.configure");
    assert(request.get("params")->at(0)->at(0)->s=="proxima.aligned-tail.v1");
    if(closes) { sock_close(fd); fd=accept_one(); }
    auto subscribe=json(line(fd)), authorize=json(line(fd));
    assert(subscribe.get("method")->s=="mining.subscribe");
    assert(authorize.get("method")->s=="mining.authorize");
    assert(authorize.get("params")->at(0)->s==sh.user);
    if(!ack.empty() && !closes && !late_ack) sendline(fd,ack);
    sendline(fd,"{\"id\":1,\"result\":[[],\"a1b2\",6],\"error\":null}");
    sendline(fd,"{\"id\":2,\"result\":true,\"error\":null}");
    sendline(fd,"{\"method\":\"mining.set_difficulty\",\"params\":[8]}");
    await([&]{std::lock_guard<std::mutex> l(sh.wmu);return sh.subscribed;});
    { std::lock_guard<std::mutex> l(sh.wmu); assert(sh.en1==0xa1b2 && sh.aligned_tail_accepted==(accepted && !late_ack)); }
    // Switching layouts never changes the nonce split or share serialization.
    unsigned job_id=0xabc;
    for(int plen : {369,504,632,375}) {
        std::vector<uint8_t> prefix(plen,0x53);
        if(plen%128==120) std::fill(prefix.end()-120,prefix.end(),0);
        std::string jid=std::to_string(job_id++);
        sendline(fd,"{\"method\":\"mining.notify\",\"params\":[\""+jid+"\",\"txid-blake2b\",\""+hexs(prefix.data(),prefix.size())+"\","+std::to_string(plen)+",230000,10000,true]}");
        Job j;
        await([&]{std::lock_guard<std::mutex> l(sh.mu);if(sh.job.jid!=jid)return false;j=sh.job;return true;});
        assert(j.txid && j.hprefix==prefix && j.en1==0xa1b2 && j.k==3);
        txid::Params p;txid::prepare(prefix.data(),plen,j.k,p);
        assert(txid::is_zero_tail(p)==(plen%128==120));
        uint64_t n=0xa1b2000000000000ULL;uint8_t h[32];
        do { txid::hash_nonce(prefix,n,h); if(txid::work_bits(h)>=j.k) break; ++n; } while(true);
        txid::HitRec hit{n,txid::load64le(h+24)};
        txid_found(sh,j,&hit,1,0);
        auto submission=json(line(fd));
        assert(submission.get("method")->s=="mining.submit");
        const JV *args=submission.get("params");assert(args->a.size()==4);
        assert(args->at(0)->s==sh.user && args->at(1)->s==jid);
        char en2[16];snprintf(en2,sizeof en2,"%012llx",(unsigned long long)(n&0xffffffffffffULL));
        assert(args->at(2)->s==en2 && args->at(3)->s==hexs(h,32));
        unsigned id=(unsigned)submission.get("id")->n;assert(id>=100);
        sendline(fd,"{\"id\":"+std::to_string(id)+",\"result\":true,\"error\":null}");
        await([&]{std::lock_guard<std::mutex> l(sh.wmu);return sh.pending.empty();});
        if(late_ack && plen==369) {
            // An acknowledgement never changes the already-issued job. The
            // pool applies the layout preference to a later job, independently
            // for every connection (the same wire IDs are used by each session).
            sendline(fd,ack);
            await([&]{std::lock_guard<std::mutex> l(sh.wmu);return sh.aligned_tail_accepted;});
            std::lock_guard<std::mutex> l(sh.mu);
            assert(sh.job.jid==jid && sh.job.hprefix==prefix);
        }
    }
    sh.stop=true;shutdown(fd,SHUT_RDWR);sock_close(fd);client.join();sock_close(listener);
    assert(!sh.aligned_tail_accepted && sh.job.id==0 && sh.invalid==0);
}
int main() {
    session("{\"id\":3,\"result\":{\"proxima.aligned-tail.v1\":true},\"error\":null}",true,false);
    session("{\"id\":3,\"result\":{\"proxima.aligned-tail.v1\":false},\"error\":null}",false,false);
    session("{\"id\":3,\"result\":null,\"error\":[-32601,\"Method not found\",null]}",false,false);
    session("{\"id\":3,\"result\":{\"proxima.aligned-tail.v1\":\"true\"},\"error\":null}",false,false);
    session("{\"id\":3,\"result\":{\"proxima.aligned-tail.v1\":true},\"error\":[1,\"denied\",null]}",false,false);
    session("",false,false);
    session("",false,true);
    session("{\"id\":3,\"result\":{\"proxima.aligned-tail.v1\":true},\"error\":null}",true,false,true);
    puts("PASS: capability accept/decline/error/malformed/absent/legacy disconnect/late acknowledgement; 32 CPU-verified shares over normal/aligned layout transitions with shared wire IDs across connections; reconnect state cleared");
}
