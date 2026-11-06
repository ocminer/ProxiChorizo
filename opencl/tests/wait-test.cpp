// ProxiChorizo License 1.0; see LICENSE.
#define main miner_entry
#include "../proxichorizo_cl.cpp"
#undef main
int main(int argc,char**argv) {
 if(argc!=4||std::string(argv[1])!="--gpus")return 2;
 enumerate_devices();int dev=atoi(argv[2]);if(dev<0||dev>=(int)g_devs.size())return 2;
 cl_int err;ClDev&d=g_devs[dev];bool fail=std::string(argv[3])=="failure";
 for(int i=0;i<(fail?1:100);i++) {
  cl_event event=clCreateUserEvent(d.ctx,&err);CL_CHECK(err);
  CL_CHECK(clEnqueueBarrierWithWaitList(d.q,1,&event,nullptr));
  CL_CHECK(clRetainEvent(event));mining_events[d.q].push_back(event);
  std::thread thread([&]{std::this_thread::sleep_for(std::chrono::milliseconds(i%3));CL_CHECK(clSetUserEventStatus(event,fail?-7:CL_COMPLETE));});
  cl_int rc=mining_wait(d.q);thread.join();clReleaseEvent(event);
  if(fail?(rc>=0):(rc!=CL_SUCCESS)){fprintf(stderr,"unexpected completion status %d\n",rc);return 1;}
 }
 printf("PASS callback %s on %s\n",fail?"negative execution status":"100 immediate/delayed completions",d.name.c_str());
}
