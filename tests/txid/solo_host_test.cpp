// ProxiChorizo License 1.0; see LICENSE.
// Exercises the real shared CUDA/OpenCL host path without a GPU. Run after
// `cd opencl && python3 gen_host.py`; compile with g++ -std=c++17 -pthread -ldl.
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
#include "../../txid_blake2b.h"
#include "../../opencl/host_common.inc"

int main() {
    Shared sh; sh.have_key = false; sh.en1 = 0xa1b2;
    std::vector<uint8_t> prefix(369, 0x42);
    auto job = [&](std::string extra = "") { return "{\"id\":17,\"version\":2,\"message\":\"" + hexs(prefix.data(), prefix.size()) + "\",\"nonce_offset\":369,\"k\":0,\"ttl_ms\":10000" + extra + "}"; };
    install_job(sh, job()); assert(sh.job.txid && sh.job.has_ttl && sh.job.id == 17);
    uint8_t tight[26]; memset(tight, 0x11, 26); memcpy(sh.job.beat,tight,26);sh.job.has_beat=true;
    install_job(sh, job(",\"beat\":\""+std::string(52,'f')+"\""));
    assert(sh.job.has_beat && !memcmp(sh.job.beat,tight,26));
    auto invalid = [&](std::string body) { install_job(sh,body); assert(sh.job.id==0); };
    auto change = [&](const std::string &from,const std::string &to) { auto s=job();s.replace(s.find(from),from.size(),to);return s; };
    invalid(change("\"nonce_offset\":369","\"nonce_offset\":368"));
    invalid(change("\"nonce_offset\":369,",""));
    invalid(change("\"k\":0","\"k\":65"));
    invalid(change("\"k\":0,",""));
    invalid(change("\"version\":2","\"version\":3"));
    invalid(job(",\"beat\":\"abcd\""));
    install_job(sh,change("\"ttl_ms\":10000","\"ttl_ms\":0"));
    assert(sh.job.has_ttl && sh.job.deadline<=std::chrono::steady_clock::now());
    install_job(sh,job());sh.job.has_beat=false;
    // These are real CPU-valid hits. The old pool cap would drop all of them
    // after 256 candidates, despite the solo server still accepting improvement.
    std::vector<txid::HitRec> hits;uint64_t best_nonce=0;uint8_t best[32];memset(best,0xff,32);
    for(unsigned i=0;i<32;i++) {
        uint64_t n=((uint64_t)sh.en1<<48)|i;uint8_t h[32];txid::hash_nonce(prefix,n,h);
        hits.push_back({n,txid::load64le(h+24)});
        int order=memcmp(h+24,best+24,8);if(order<0||(order==0&&memcmp(h+6,best+6,18)<0)){memcpy(best,h,32);best_nonce=n;}
    }
    sh.cap_job=17;sh.cap_n=256;sh.cap_best=64;
    int fd=socket(AF_INET,SOCK_STREAM,0);assert(fd>=0);sockaddr_in addr{};addr.sin_family=AF_INET;addr.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
    assert(bind(fd,(sockaddr*)&addr,sizeof addr)==0);assert(listen(fd,4)==0);socklen_t sz=sizeof addr;assert(getsockname(fd,(sockaddr*)&addr,&sz)==0);
    assert(parse_url("http://127.0.0.1:"+std::to_string(ntohs(addr.sin_port)),sh.url));
    std::atomic<int> requests{0};std::string request;
    std::thread server([&] {
        fd_set set;FD_ZERO(&set);FD_SET(fd,&set);timeval timeout{3,0};
        if(select(fd+1,&set,nullptr,nullptr,&timeout)<=0)return;
        int c=accept(fd,nullptr,nullptr);sock_timeout(c,2);char buf[4096];auto got=recv(c,buf,sizeof buf,0);assert(got>0);request.assign(buf,got);requests++;
        std::string beat=hexs(best+24,8)+hexs(best+6,18);
        std::string body="{\"accepted\":true,\"beat\":\""+beat+"\"}";
        char len[32];snprintf(len,sizeof len,"%zx",body.size());
        std::string out="HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"+std::string(len)+"\r\n"+body+"\r\n0\r\n\r\n";
        assert(send(c,out.data(),out.size(),MSG_NOSIGNAL)==(ssize_t)out.size());sock_close(c);
    });
    txid_found(sh,sh.job,hits.data(),hits.size(),0);server.join();sock_close(fd);
    assert(requests==1 && sh.solutions==1);uint8_t nb[8];for(int i=0;i<8;i++)nb[i]=(uint8_t)(best_nonce>>(56-8*i));
    assert(request.find("\"nonce\":\""+hexs(nb,8)+"\"")!=std::string::npos);
    assert(request.find("\"beta\":\""+hexs(best,32)+"\"")!=std::string::npos);
    auto bad=hits.front();bad.h3^=1;txid_found(sh,sh.job,&bad,1,0);assert(sh.invalid==1);
    // Pool credit limiting remains unchanged.
    sh.url.stratum=true;txid_found(sh,sh.job,hits.data(),hits.size(),0);assert(sh.cap_n==256);
    puts("PASS: late solo improvement, best work-key order, CPU verification, pool cap, job validation, zero TTL, monotone beat, chunked HTTP");
}
