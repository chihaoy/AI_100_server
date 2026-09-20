// 一个 program 两个 specialization:用 setDataExt 给 buffer 维度来选形状,交替执行计时
#include "QAicApi.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
using clk=std::chrono::steady_clock;
static double ms(clk::time_point a,clk::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
static double med(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}
int main(int argc,char**argv){
  std::vector<QID> devs{0}; QAicContext*ctx=nullptr; qaicCreateContext(&ctx,nullptr,1,devs.data(),nullptr,nullptr,nullptr,nullptr);
  QAicQpcObj*qpc=nullptr; qaicOpenQpcFile(&qpc,(std::string(argv[1])+"/programqpc.bin").c_str());
  QAicProgram*prog=nullptr; QAicProgramProperties_t pp; qaicProgramPropertiesInitDefault(&pp);
  qaicCreateProgram(ctx,&prog,&pp,0,"spec",qpc); qaicLoadProgram(prog); qaicRunActivationCmd(prog,QAIC_PROGRAM_CMD_ACTIVATE_FULL);
  QAicQueue*q=nullptr; qaicCreateQueue(ctx,&q,nullptr,0);
  QData iod{}; qaicProgramGetIoDescriptor(prog,&iod); QAicExecObjProperties_t ep=QAIC_EXECOBJ_PROPERTIES_DEFAULT; QAicExecObj*eo=nullptr;
  if(qaicCreateExecObj(ctx,&eo,&ep,prog,&iod,nullptr,nullptr)!=QS_SUCCESS){printf("execobj fail\n");return 1;}
  QAicIoBufferInfo_t*bi=nullptr; qaicProgramGetIoBufferInfo(prog,&bi); uint32_t nb=bi->numBufferMappings;
  std::vector<QBuffer> bufs(nb); for(uint32_t k=0;k<nb;k++){auto&m=bi->bufferMappings[k];QBuffer&b=bufs[m.index];b.size=m.size;b.buf=(uint8_t*)aligned_alloc(4096,(m.size+4095)/4096*4096);b.type=QBUFFER_TYPE_HEAP;printf("  buffer[%u] %s size(max) %u\n",m.index,m.bufferName,m.size);}
  auto run=[&](uint32_t T){
    std::vector<QBufferDimensions> dims(nb);
    for(uint32_t k=0;k<nb;k++){dims[k].sizeOfElem=2; dims[k].count=2; dims[k].dims=new uint32_t[2]{T,2048}; bufs[k].size=T*2048*2;}   // buffer 大小也要和本次形状一致   // x 和 y 都是 [T,2048] fp16
    if(qaicExecObjSetDataExt(eo,nb,bufs.data(),dims.data())!=QS_SUCCESS){printf("setDataExt T=%u FAIL\n",T);exit(1);}
    qaicEnqueueExecObj(q,eo,nullptr); qaicExecObjWaitForCompletion(eo,60000000);
  };
  for(int i=0;i<3;i++){run(128);run(16);}
  std::vector<double> t128,t16,alt;
  for(int i=0;i<20;i++){auto a=clk::now();run(128);t128.push_back(ms(a,clk::now()));}
  for(int i=0;i<20;i++){auto a=clk::now();run(16);t16.push_back(ms(a,clk::now()));}
  std::vector<double> a128,a16;
  for(int i=0;i<40;i++){auto a=clk::now();run(i%2?16:128);(i%2?a16:a128).push_back(ms(a,clk::now()));}
  printf("同一个 program,按 buffer 维度选 specialization(权重同一份):\n  seq_len=128: 单独反复 %.2f ms, 交替时 %.2f ms\n  seq_len=16:  单独反复 %.2f ms, 交替时 %.2f ms\n  -> 形状切换开销 ≈ %.2f ms\n",med(t128),med(a128),med(t16),med(a16),((med(a128)-med(t128))+(med(a16)-med(t16)))/2);
  return 0;
}
