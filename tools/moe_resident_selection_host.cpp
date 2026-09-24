// Fixed-shape expert-ID values select resident weights. No program reload in loop.
#include "QAicApi.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

static void check(QStatus s,const char *what) {
  if(s!=QS_SUCCESS)throw std::runtime_error(std::string(what)+": "+std::to_string(s));
}
#define CK(x) check(x,#x)
using Clock=std::chrono::steady_clock;
static std::vector<uint8_t> read(const std::string &p) {
  std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("Cannot read "+p);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),{});
}
static void write(const std::string &p,const QBuffer &b) {
  std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(b.buf),b.size);
  if(!f)throw std::runtime_error("Cannot write "+p);
}
struct Session {
  QAicContext *ctx=nullptr;QAicQueue *queue=nullptr;QAicQpcObj *qpc=nullptr;
  QAicProgram *program=nullptr;QAicExecObj *exec=nullptr;bool loaded=false,active=false;
  std::vector<QBuffer> buffers;
  ~Session() {
    if(exec)qaicReleaseExecObj(exec);
    if(active)qaicRunActivationCmd(program,QAIC_PROGRAM_CMD_DEACTIVATE_FULL);
    if(loaded)qaicUnloadProgram(program);
    if(program)qaicReleaseProgram(program);
    if(qpc)qaicCloseQpc(qpc);
    if(queue)qaicReleaseQueue(queue);
    if(ctx)qaicReleaseContext(ctx);
    for(auto &b:buffers)free(b.buf);
  }
};

int main(int argc,char **argv) try {
  if(argc!=8)throw std::runtime_error("usage: host QPC_DIR INPUT_DIR OUT_DIR DEVICE H ITERATIONS VARIANT");
  std::string qpcdir=argv[1],root=argv[2],out=argv[3],variant=argv[7];
  QID device=std::stoi(argv[4]);uint32_t H=std::stoul(argv[5]);int N=std::stoi(argv[6]);
  bool dynamic=variant=="dynamic";
  const std::array<std::string,4> names={"identity","swap","shuffle","duplicate_control"};
  int fixed_case=-1;
  for(int i=0;i<3;i++)if(variant=="static_"+names[i])fixed_case=i;
  if((!dynamic&&fixed_case<0)||H<128||N<1||std::filesystem::exists(out))throw std::runtime_error("Invalid arguments");
  std::filesystem::create_directories(out);
  auto x=read(root+"/x.bin");if(x.size()!=128*H*2)throw std::runtime_error("Invalid activation size");
  std::array<std::vector<uint8_t>,4> orders;
  for(int i=0;i<4;i++) {
    orders[i]=read(root+"/orders/"+names[i]+".bin");
    if(orders[i].size()!=128)throw std::runtime_error("Invalid expert ID size");
    std::array<int32_t,32> ids{};std::memcpy(ids.data(),orders[i].data(),128);
    for(auto id:ids)if(id<0||id>=32)throw std::runtime_error("Expert ID outside resident bank");
    if(i<3) {std::sort(ids.begin(),ids.end());for(int e=0;e<32;e++)if(ids[e]!=e)throw std::runtime_error("Invalid permutation");}
  }
  std::ofstream resources(out+"/resources.csv");resources<<"stage,dram_free_kib,nsp_free\n";
  auto snapshot=[&](const std::string &stage) {
    QResourceInfo info{};CK(qaicGetResourceInfo(device,&info));
    resources<<stage<<","<<info.dramFree<<","<<unsigned(info.nspFree)<<"\n";resources.flush();
  };
  snapshot("before_load");Session s;
  CK(qaicCreateContext(&s.ctx,nullptr,1,&device,nullptr,nullptr,nullptr,nullptr));
  CK(qaicCreateQueue(s.ctx,&s.queue,nullptr,device));
  CK(qaicOpenQpcFile(&s.qpc,(qpcdir+"/programqpc.bin").c_str()));
  const QAicQpcInfo_t *qi=nullptr;CK(qaicQpcGetInfo(s.qpc,&qi));
  if(qi->numPrograms!=1)throw std::runtime_error("Expected one program");
  uint64_t constants_bytes=0;for(uint32_t i=0;i<qi->numConstants;i++)constants_bytes+=qi->constantsInfo[i].size;
  QAicProgramProperties_t pp;CK(qaicProgramPropertiesInitDefault(&pp));
  CK(qaicCreateProgram(s.ctx,&s.program,&pp,device,"resident_selection",s.qpc));
  CK(qaicLoadProgram(s.program));s.loaded=true;
  CK(qaicRunActivationCmd(s.program,QAIC_PROGRAM_CMD_ACTIVATE_FULL));s.active=true;snapshot("activated");
  QData iod{};CK(qaicProgramGetIoDescriptor(s.program,&iod));
  QAicExecObjProperties_t ep=QAIC_EXECOBJ_PROPERTIES_DEFAULT;
  CK(qaicCreateExecObj(s.ctx,&s.exec,&ep,s.program,&iod,nullptr,nullptr));
  QAicIoBufferInfo_t *bi=nullptr;CK(qaicProgramGetIoBufferInfo(s.program,&bi));
  s.buffers.resize(bi->numBufferMappings);int yi=-1,ci=-1,ii=-1;bool has_x=false;
  for(uint32_t k=0;k<bi->numBufferMappings;k++) {
    auto &m=bi->bufferMappings[k];if(m.index>=s.buffers.size())throw std::runtime_error("Invalid buffer index");
    auto &b=s.buffers[m.index];b.size=m.size;b.type=QBUFFER_TYPE_HEAP;
    b.buf=static_cast<uint8_t*>(aligned_alloc(4096,(b.size+4095)/4096*4096));if(!b.buf)throw std::bad_alloc();
    std::memset(b.buf,0,b.size);std::string name=m.bufferName;
    if(name=="x") {if(b.size!=x.size())throw std::runtime_error("x size mismatch");std::memcpy(b.buf,x.data(),x.size());has_x=true;}
    else if(name=="y") {yi=m.index;if(b.size!=x.size())throw std::runtime_error("y size mismatch");}
    else if(name=="counts") {ci=m.index;if(b.size!=128)throw std::runtime_error("counts size mismatch");}
    else if(name=="expert_ids") {ii=m.index;if(b.size!=128)throw std::runtime_error("ID size mismatch");}
    else throw std::runtime_error("Unexpected buffer "+name);
    std::cout<<name<<" bytes="<<b.size<<" dtype="<<m.dataType<<"\n";
  }
  if(!has_x||yi<0||ci<0||dynamic!=(ii>=0))throw std::runtime_error("Unexpected interface");
  std::array<std::vector<uint8_t>,4> expected_y,expected_counts;
  auto run=[&](int index) {
    auto start=Clock::now();
    if(dynamic)std::memcpy(s.buffers[ii].buf,orders[index].data(),orders[index].size());
    CK(qaicExecObjSetData(s.exec,s.buffers.size(),s.buffers.data()));
    CK(qaicEnqueueExecObj(s.queue,s.exec,nullptr));CK(qaicExecObjWaitForCompletion(s.exec,60000000));
    double elapsed=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    if(!expected_y[index].empty() &&
       (std::memcmp(s.buffers[yi].buf,expected_y[index].data(),expected_y[index].size()) ||
        std::memcmp(s.buffers[ci].buf,expected_counts[index].data(),expected_counts[index].size())))
      throw std::runtime_error("Nondeterministic output: "+names[index]);
    return elapsed;
  };
  for(int i=0;i<4;i++)if(dynamic||i==fixed_case) {
    run(i);expected_y[i].assign(s.buffers[yi].buf,s.buffers[yi].buf+s.buffers[yi].size);
    expected_counts[i].assign(s.buffers[ci].buf,s.buffers[ci].buf+s.buffers[ci].size);
    write(out+"/"+names[i]+"_y.bin",s.buffers[yi]);write(out+"/"+names[i]+"_counts.bin",s.buffers[ci]);snapshot(names[i]);
  }
  std::ofstream csv(out+"/samples.csv");csv<<std::setprecision(9)<<"round,mode,case,total_ms\n";
  auto sample=[&](int round,const char *mode,int i) {csv<<round<<","<<mode<<","<<names[i]<<","<<run(i)<<"\n";};
  for(int round=0;round<3;round++) {
    auto solo=[&]() {for(int j=0;j<3;j++) {int i=(round%2)?2-j:j;if(!dynamic&&i!=fixed_case)continue;
      for(int w=0;w<10;w++)run(i);
      for(int n=0;n<N;n++)sample(round,"solo",i);
    }};
    auto cycle=[&]() {if(!dynamic)return;for(int w=0;w<30;w++)run(w%3);
      for(int n=0;n<N*3;n++)sample(round,"cycle",(round%2)?2-n%3:n%3);
    };
    if(round%2){cycle();solo();}else{solo();cycle();}
  }
  csv.close();if(!csv)throw std::runtime_error("Failed writing samples");
  std::ofstream meta(out+"/metadata.json");meta<<"{\"program_loads\":1,\"program_activations\":1,\"exec_objects\":1,"
    <<"\"num_programs\":"<<qi->numPrograms<<",\"constants_bytes\":"<<constants_bytes
    <<",\"program_req_mem_bytes\":"<<qi->programInfo[0].reqMem<<",\"device\":"<<device<<"}\n";
  meta.close();if(!meta)throw std::runtime_error("Failed writing metadata");
  std::cout<<"ALL_REPETITIONS_EXACT\n";return 0;
} catch(const std::exception &e) {std::cerr<<"ERROR "<<e.what()<<"\n";return 1;}
