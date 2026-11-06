// ProxiChorizo License 1.0; see LICENSE.
#define main miner_entry
#include "../proxichorizo_cl.cpp"
#undef main
#include <array>
struct Row { std::array<uint8_t,64> raw,expected;int valid; };
int main(int argc,char**argv) {
    if(argc!=5 || std::string(argv[1])!="--gpus")return 2;
    g_kernel_src.assign((const char*)CL_SRC,CL_SRC_LEN);
    g_kernel_src+=R"CL(
__kernel void point_load(__global const u8 *raw,__global HPoint *pts,uint n) {
 uint i=get_global_id(0);if(i>=n)return;u8 x[32],y[32];
 for(int j=0;j<32;j++){x[j]=raw[64*i+j];y[j]=raw[64*i+32+j];}
 fe X,Y;fe_frombytes(&X,x);fe_frombytes(&Y,y);pts[i].X=X;pts[i].Y=Y;pts[i].nonce=i;
}
__kernel void point_dump(__global const ProjectivePoint *pts,__global u8 *out,__global uint *valid,uint n) {
 uint i=get_global_id(0);if(i>=n)return;valid[i]=pts[i].valid;if(!pts[i].valid)return;
 fe x=pts[i].X,y=pts[i].Y,z=pts[i].Z,zi;fe_invert(&zi,&z);fe_mul(&x,&x,&zi);fe_mul(&y,&y,&zi);
 u8 xb[32],yb[32];fe_tobytes(xb,&x);fe_tobytes(yb,&y);
 for(int j=0;j<32;j++){out[64*i+j]=xb[j];out[64*i+32+j]=yb[j];}
}
)CL";
    enumerate_devices();int dev=atoi(argv[2]);if(dev<0||dev>=(int)g_devs.size())return 2;
    const Variant*v=variantByName(argv[3]);if(!v||!v->split)return 2;
    std::ifstream in(argv[4]);std::map<std::string,std::vector<Row>> groups;std::string sc,x,y,ex,ey;int valid;
    while(in>>sc>>x>>y>>ex>>ey>>valid) {
        Row r;std::vector<uint8_t>b;if(!unhex(x+y,b)||b.size()!=64)return 2;std::copy(b.begin(),b.end(),r.raw.begin());
        if(!unhex(ex+ey,b)||b.size()!=64)return 2;std::copy(b.begin(),b.end(),r.expected.begin());r.valid=valid;groups[sc].push_back(r);
    }
    if(groups.size()!=10)return 2;Prog*pr=get_prog(g_devs[dev],v);ClDev&d=*pr->dev;cl_int e;int tested=0;
    cl_kernel load=clCreateKernel(pr->p,"point_load",&e);CL_CHECK(e);cl_kernel dump=clCreateKernel(pr->p,"point_dump",&e);CL_CHECK(e);
    for(const auto &group:groups) {
        std::vector<uint8_t>scalar;unhex(group.first,scalar);uint8_t pk[32];if(prog_init(pr,scalar.data(),pk)!=1)return 3;
        for(cl_uint n:{1u,17u,513u,4097u}) {
            std::vector<uint8_t>raw(64*n),out(64*n);std::vector<cl_uint>ok(n);
            for(unsigned i=0;i<n;i++)std::copy(group.second[i%group.second.size()].raw.begin(),group.second[i%group.second.size()].raw.end(),raw.begin()+64*i);
            Scratch*s=scratch_alloc(pr,n);
            cl_mem input=clCreateBuffer(d.ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,raw.size(),raw.data(),&e);CL_CHECK(e);
            cl_mem output=clCreateBuffer(d.ctx,CL_MEM_READ_WRITE,out.size(),nullptr,&e);CL_CHECK(e);
            cl_mem flags=clCreateBuffer(d.ctx,CL_MEM_READ_WRITE,n*4,nullptr,&e);CL_CHECK(e);
            CL_CHECK(clEnqueueWriteBuffer(d.q,s->cnt,CL_TRUE,0,4,&n,0,nullptr,nullptr));
            size_t l=256,g=((n+255)/256)*256;
            CL_CHECK(clSetKernelArg(load,0,sizeof input,&input));CL_CHECK(clSetKernelArg(load,1,sizeof s->pts,&s->pts));CL_CHECK(clSetKernelArg(load,2,sizeof n,&n));
            CL_CHECK(clEnqueueNDRangeKernel(d.q,load,1,nullptr,&g,&l,0,nullptr,nullptr));
            if(v->fast) {
                size_t pg=256;
                CL_CHECK(clSetKernelArg(pr->k_prep,0,sizeof s->cnt,&s->cnt));CL_CHECK(clSetKernelArg(pr->k_prep,1,sizeof s->pts,&s->pts));CL_CHECK(clSetKernelArg(pr->k_prep,2,sizeof pr->consts,&pr->consts));
                CL_CHECK(clEnqueueNDRangeKernel(d.q,pr->k_prep,1,nullptr,&pg,&l,0,nullptr,nullptr));
            }
            CL_CHECK(clSetKernelArg(pr->k_scalar,0,sizeof s->pts,&s->pts));CL_CHECK(clSetKernelArg(pr->k_scalar,1,sizeof s->cnt,&s->cnt));CL_CHECK(clSetKernelArg(pr->k_scalar,2,sizeof s->projective,&s->projective));CL_CHECK(clSetKernelArg(pr->k_scalar,3,sizeof pr->consts,&pr->consts));
            CL_CHECK(clEnqueueNDRangeKernel(d.q,pr->k_scalar,1,nullptr,&g,&l,0,nullptr,nullptr));
            CL_CHECK(clSetKernelArg(dump,0,sizeof s->projective,&s->projective));CL_CHECK(clSetKernelArg(dump,1,sizeof output,&output));CL_CHECK(clSetKernelArg(dump,2,sizeof flags,&flags));CL_CHECK(clSetKernelArg(dump,3,sizeof n,&n));
            CL_CHECK(clEnqueueNDRangeKernel(d.q,dump,1,nullptr,&g,&l,0,nullptr,nullptr));
            CL_CHECK(clEnqueueReadBuffer(d.q,output,CL_TRUE,0,out.size(),out.data(),0,nullptr,nullptr));CL_CHECK(clEnqueueReadBuffer(d.q,flags,CL_TRUE,0,n*4,ok.data(),0,nullptr,nullptr));
            for(unsigned i=0;i<n;i++) {
                const Row&r=group.second[i%group.second.size()];int off=v->fast?32:0;
                if(ok[i]!=(unsigned)r.valid || (r.valid&&memcmp(out.data()+64*i+off,r.expected.data()+off,64-off))) {fprintf(stderr,"POINT FAIL scalar=%s n=%u i=%u valid=%u expected=%d\n",group.first.c_str(),n,i,ok[i],r.valid);return 1;}
                tested++;
            }
            scratch_free(s);clReleaseMemObject(input);clReleaseMemObject(output);clReleaseMemObject(flags);
        }
    }
    printf("PASS %s: %d independent point outputs, ten scalars/all torsion/partial batches/Q=-P\n",v->name,tested);
}
