// 测:ProgramGroup 能不能让多个 program 同时常驻一张卡,组内切换是否便宜
#include "QAicApi.h"
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) { return std::chrono::duration<double,std::milli>(b-a).count(); }
int main(int argc, char** argv) {
  int n = argc - 1;
  std::vector<QID> devs{0,1};
  QAicContext* ctx=nullptr;
  if (qaicCreateContext(&ctx,nullptr,devs.size(),devs.data(),nullptr,nullptr,nullptr,nullptr)!=QS_SUCCESS){printf("ctx fail\n");return 1;}
  QAicProgramGroup* pg=nullptr; QAicQueue* q=nullptr;
  QAicProgramGroupProperties_t pgp{}; qaicProgramGroupPropertiesInitDefault(&pgp);
  QStatus s = qaicCreateProgramGroup(ctx,&pg,&q,&pgp,"pgtest",0);
  printf("createProgramGroup: %d %s\n",(int)s, s==QS_SUCCESS?"OK":"FAIL");
  if (s!=QS_SUCCESS) return 1;
  std::vector<QAicProgram*> progs;
  for (int i=0;i<n;i++){
    QAicQpcObj* qpc=nullptr;
    std::string p = std::string(argv[i+1])+"/programqpc.bin";
    if (qaicOpenQpcFile(&qpc,p.c_str())!=QS_SUCCESS){printf("open %s fail\n",p.c_str());return 1;}
    QAicProgram* prog=nullptr;                       // addProgram 自己创建 program
    QAicProgramProperties_t pp; qaicProgramPropertiesInitDefault(&pp); pp.devMapping="0:1";
    auto t2=clk::now();
    QStatus as = qaicProgramGroupAddProgram(pg,qpc,&prog,&pp);
    auto t3=clk::now();
    printf("  [%d] addProgram: %d %s (%.0f ms)\n",i,(int)as,as==QS_SUCCESS?"OK":"FAIL",ms(t2,t3));
    if (as!=QS_SUCCESS) return 1;
    progs.push_back(prog);
  }
  auto t0=clk::now(); QStatus es = qaicProgramGroupEnable(pg); auto t1=clk::now();
  printf("enable (%d programs): %d %s (%.0f ms)  <- 若成功说明多个 program 可同时常驻\n",n,(int)es,es==QS_SUCCESS?"OK":"FAIL",ms(t0,t1));
  return es==QS_SUCCESS?0:1;
}
