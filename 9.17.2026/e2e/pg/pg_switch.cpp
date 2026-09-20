// ProgramGroup 常驻多个单卡 program,交替 enqueue,量切换成本
#include "QAicApi.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
using clk=std::chrono::steady_clock;
static double ms(clk::time_point a,clk::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
static void rep(const char*n,std::vector<double> t){std::sort(t.begin(),t.end());size_t m=t.size();printf("  %-40s median %8.3f ms  p10 %8.3f  p90 %8.3f\n",n,t[m/2],t[m/10],t[9*m/10]);}
int main(int argc,char**argv){
  int n=argc-1; std::vector<QID> devs{0}; QAicContext*ctx=nullptr;
  qaicCreateContext(&ctx,nullptr,1,devs.data(),nullptr,nullptr,nullptr,nullptr);
  QAicProgramGroup*pg=nullptr; QAicQueue*q=nullptr; QAicProgramGroupProperties_t pgp{}; qaicProgramGroupPropertiesInitDefault(&pgp);
  if(qaicCreateProgramGroup(ctx,&pg,&q,&pgp,"sw",0)!=QS_SUCCESS){printf("group fail\n");return 1;}
  std::vector<QAicProgram*> progs; std::vector<QAicExecObj*> eos; std::vector<std::vector<QBuffer>> bufs(n);
  auto t0=clk::now();
  for(int i=0;i<n;i++){
    QAicQpcObj*qpc=nullptr; std::string p=std::string(argv[i+1])+"/programqpc.bin";
    if(qaicOpenQpcFile(&qpc,p.c_str())!=QS_SUCCESS){printf("open fail %s\n",p.c_str());return 1;}
    QAicProgram*prog=nullptr; QAicProgramProperties_t pp; qaicProgramPropertiesInitDefault(&pp);
    QStatus s=qaicProgramGroupAddProgram(pg,qpc,&prog,&pp);
    if(s!=QS_SUCCESS){printf("addProgram[%d] FAIL %d  (只能加 %d 个)\n",i,(int)s,i);return 1;}
    progs.push_back(prog);
  }
  auto t1=clk::now();
  QStatus es=qaicProgramGroupEnable(pg); auto t2=clk::now();
  printf("%d 个单卡 program: add 共 %.0f ms, enable %.0f ms -> %s\n",n,ms(t0,t1),ms(t1,t2),es==QS_SUCCESS?"全部常驻":"FAIL");
  if(es!=QS_SUCCESS) return 1;
  for(int i=0;i<n;i++){
    QData iod{}; qaicProgramGetIoDescriptor(progs[i],&iod);
    QAicExecObjProperties_t ep=QAIC_EXECOBJ_PROPERTIES_DEFAULT; QAicExecObj*eo=nullptr;
    if(qaicCreateExecObj(ctx,&eo,&ep,progs[i],&iod,nullptr,nullptr)!=QS_SUCCESS){printf("execobj[%d] fail\n",i);return 1;}
    QAicIoBufferInfo_t*bi=nullptr; qaicProgramGetIoBufferInfo(progs[i],&bi);
    bufs[i].assign(bi->numBufferMappings,QBuffer{});
    for(uint32_t k=0;k<bi->numBufferMappings;k++){auto&m=bi->bufferMappings[k];QBuffer&b=bufs[i][m.index];b.size=m.size;b.buf=(uint8_t*)aligned_alloc(4096,(m.size+4095)/4096*4096);b.type=QBUFFER_TYPE_HEAP;}
    qaicExecObjSetData(eo,bufs[i].size(),bufs[i].data()); eos.push_back(eo);
  }
  auto run=[&](int i){qaicEnqueueExecObj(q,eos[i],nullptr);qaicExecObjWaitForCompletion(eos[i],60000000);};
  for(int r=0;r<3;r++) for(int i=0;i<n;i++) run(i);               // warm
  printf("常驻后,每个 program 单独反复跑(无切换):\n");
  double solo_sum=0;
  for(int i=0;i<n;i++){std::vector<double> t;for(int k=0;k<20;k++){auto a=clk::now();run(i);t.push_back(ms(a,clk::now()));}
    std::sort(t.begin(),t.end()); solo_sum+=t[10]; printf("  [%d] %-28s median %8.3f ms\n",i,argv[i+1],t[10]);}
  std::vector<double> alt;
  for(int k=0;k<20*n;k++){auto a=clk::now();run(k%n);alt.push_back(ms(a,clk::now()));}
  std::sort(alt.begin(),alt.end());
  double alt_mean=0; for(double v:alt) alt_mean+=v; alt_mean/=alt.size();
  printf("轮流切换着跑: 每次平均 %.3f ms;  无切换时这 %d 个的平均 %.3f ms  -> 每次切换开销 %.3f ms\n",
         alt_mean,n,solo_sum/n,alt_mean-solo_sum/n);
  return 0;
}
