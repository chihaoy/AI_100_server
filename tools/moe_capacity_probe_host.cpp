// One program/ExecObj, shape-selected capacities, fixed token input.
// Checks every normal execution outside the timed region; explicit error cleanup.
#include "QAicApi.h"
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(QStatus s, const char *what) {
  if (s != QS_SUCCESS) throw std::runtime_error(std::string(what)+": "+std::to_string(s));
}
#define CK(x) check(x, #x)
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double,std::milli>(b-a).count();
}
static std::vector<uint8_t> read(const std::string &p) {
  std::ifstream f(p,std::ios::binary);
  if (!f) throw std::runtime_error("Cannot read "+p);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),{});
}
static void write(const std::string &p, const QBuffer &b) {
  std::ofstream f(p,std::ios::binary);
  f.write(reinterpret_cast<const char*>(b.buf),b.size);
  if (!f) throw std::runtime_error("Cannot write "+p);
}
struct Session {
  QAicContext *ctx=nullptr; QAicQueue *queue=nullptr; QAicQpcObj *qpc=nullptr;
  QAicProgram *program=nullptr; QAicExecObj *exec=nullptr;
  bool loaded=false, active=false;
  std::vector<QBuffer> buffers;
  ~Session() {
    if(exec) qaicReleaseExecObj(exec);
    if(active) qaicRunActivationCmd(program,QAIC_PROGRAM_CMD_DEACTIVATE_FULL);
    if(loaded) qaicUnloadProgram(program);
    if(program) qaicReleaseProgram(program);
    if(qpc) qaicCloseQpc(qpc);
    if(queue) qaicReleaseQueue(queue);
    if(ctx) qaicReleaseContext(ctx);
    for(auto &b:buffers) free(b.buf);
  }
};

int main(int argc,char **argv) try {
  if(argc!=7) throw std::runtime_error("usage: host QPC_DIR INPUT_DIR OUT_DIR DEVICES H ITERATIONS");
  std::string qpcdir=argv[1], inputdir=argv[2], outdir=argv[3], deviceids=argv[4];
  const uint32_t H=std::stoul(argv[5]); const int N=std::stoi(argv[6]);
  if(N<1 || H<1 || std::filesystem::exists(outdir)) throw std::runtime_error("Invalid args or output exists");
  std::filesystem::create_directories(outdir);
  std::vector<QID> devices; std::stringstream ss(deviceids); std::string part, mapping;
  while(std::getline(ss,part,',')) { devices.push_back(std::stoi(part)); mapping+=(mapping.empty()?"":":")+part; }
  auto x=read(inputdir+"/x.bin"), stress=read(inputdir+"/stress_x.bin");
  if(x.size()!=128*H*2 || stress.size()!=x.size()) throw std::runtime_error("Invalid input bytes");
  std::ofstream resources(outdir+"/resources.csv"); resources<<"stage,device,dram_free_kib,nsp_free\n";
  auto resource_snapshot=[&](const std::string &stage) {
    for(auto device:devices) {
      QResourceInfo info{}; CK(qaicGetResourceInfo(device,&info));
      resources<<stage<<","<<device<<","<<info.dramFree<<","<<unsigned(info.nspFree)<<"\n";
    }
    resources.flush();
  };
  resource_snapshot("before_load");
  Session s;
  CK(qaicCreateContext(&s.ctx,nullptr,devices.size(),devices.data(),nullptr,nullptr,nullptr,nullptr));
  CK(qaicCreateQueue(s.ctx,&s.queue,nullptr,devices[0]));
  CK(qaicOpenQpcFile(&s.qpc,(qpcdir+"/programqpc.bin").c_str()));
  const QAicQpcInfo_t *qi=nullptr; CK(qaicQpcGetInfo(s.qpc,&qi));
  if(qi->numPrograms!=1) throw std::runtime_error("Expected one QPC program");
  uint64_t constants_bytes=0;
  for(uint32_t k=0;k<qi->numConstants;k++) constants_bytes+=qi->constantsInfo[k].size;
  QAicProgramProperties_t pp; CK(qaicProgramPropertiesInitDefault(&pp)); pp.devMapping=mapping.c_str();
  CK(qaicCreateProgram(s.ctx,&s.program,&pp,devices[0],"capacity_probe",s.qpc));
  CK(qaicLoadProgram(s.program)); s.loaded=true;
  CK(qaicRunActivationCmd(s.program,QAIC_PROGRAM_CMD_ACTIVATE_FULL)); s.active=true;
  resource_snapshot("after_activation");
  QData iod{}; CK(qaicProgramGetIoDescriptor(s.program,&iod));
  QAicExecObjProperties_t ep=QAIC_EXECOBJ_PROPERTIES_DEFAULT;
  CK(qaicCreateExecObj(s.ctx,&s.exec,&ep,s.program,&iod,nullptr,nullptr));
  QAicIoBufferInfo_t *bi=nullptr; CK(qaicProgramGetIoBufferInfo(s.program,&bi));
  const uint32_t nb=bi->numBufferMappings;
  s.buffers.resize(nb); std::vector<QBufferDimensions> dims(nb);
  std::vector<std::array<uint32_t,2>> shapes(nb); std::vector<std::string> names(nb);
  int xi=-1,yi=-1,ci=-1,r0=-1,r1=-1,gi=-1,ti=-1;
  for(uint32_t k=0;k<nb;k++) {
    auto &m=bi->bufferMappings[k]; uint32_t i=m.index;
    if(i>=nb) throw std::runtime_error("Invalid mapping index");
    names[i]=m.bufferName;
    auto &b=s.buffers[i]; b.size=m.size; b.type=QBUFFER_TYPE_HEAP;
    b.buf=static_cast<uint8_t*>(aligned_alloc(4096,(m.size+4095)/4096*4096));
    if(!b.buf) throw std::bad_alloc();
    std::memset(b.buf,0,b.size); dims[i].dims=shapes[i].data();
    if(names[i]=="x" || names[i]=="y") {
      shapes[i]={128,H}; dims[i].sizeOfElem=2; dims[i].count=2;
      if(b.size!=x.size()) throw std::runtime_error("Unexpected hidden state buffer size");
      if(names[i]=="x") { xi=i; std::memcpy(b.buf,x.data(),x.size()); } else yi=i;
    } else if(names[i]=="counts") {
      ci=i; shapes[i]={128,0}; dims[i].sizeOfElem=4; dims[i].count=1;
      if(b.size!=128*4) throw std::runtime_error("Unexpected counts size");
    } else if(names[i]=="tag") {
      ti=i; shapes[i]={4,0}; dims[i].sizeOfElem=4; dims[i].count=1;
      if(b.size!=4*4) throw std::runtime_error("Unexpected tag buffer size");
    } else if(names[i]=="grid") {
      gi=i; shapes[i]={128,128}; dims[i].sizeOfElem=4; dims[i].count=2;
      if(b.size!=128*128*4) throw std::runtime_error("Expected int32 joint capacity buffer");
    } else if(names[i]=="rows0" || names[i]=="rows1") {
      shapes[i]={128,0}; dims[i].sizeOfElem=4; dims[i].count=1;
      if(b.size!=128*4) throw std::runtime_error("Unexpected capacity buffer size");
      for(int j=0;j<128;j++) reinterpret_cast<int32_t*>(b.buf)[j]=j;
      if(names[i]=="rows0") r0=i; else r1=i;
    } else throw std::runtime_error("Unknown buffer "+names[i]);
    std::cout<<i<<" "<<names[i]<<" max_bytes="<<m.size<<" dtype="<<m.dataType<<"\n";
  }
  if(xi<0 || yi<0 || ci<0 || (gi<0 && (r0<0 || r1<0))) throw std::runtime_error("Missing buffer");
  const std::array<std::array<uint32_t,2>,4> profiles={{{128,128},{64,32},{32,64},{32,32}}};
  std::array<std::vector<int32_t>,4> grids;
  for(int p=0;p<4;p++) for(uint32_t i=0;i<profiles[p][0];i++)
    for(uint32_t j=0;j<profiles[p][1];j++) grids[p].push_back(i+j);
  auto setshape=[&](uint32_t c0,uint32_t c1) {
    if(gi>=0) { shapes[gi]={c0,c1}; s.buffers[gi].size=c0*c1*4; }
    else { shapes[r0][0]=c0; shapes[r1][0]=c1; s.buffers[r0].size=c0*4; s.buffers[r1].size=c1*4; }
  };
  std::vector<uint8_t> expected_y,expected_counts;
  auto run=[&](int p,bool verify=true) {
    setshape(profiles[p][0],profiles[p][1]);
    if(ti>=0) { shapes[ti][0]=p+1; s.buffers[ti].size=(p+1)*4; }
    auto a=Clock::now();
    if(gi>=0) std::memcpy(s.buffers[gi].buf,grids[p].data(),grids[p].size()*4);
    CK(qaicExecObjSetDataExt(s.exec,nb,s.buffers.data(),dims.data()));
    auto b=Clock::now(); CK(qaicEnqueueExecObj(s.queue,s.exec,nullptr));
    CK(qaicExecObjWaitForCompletion(s.exec,60000000)); auto c=Clock::now();
    if(verify && !expected_y.empty()) {
      if(std::memcmp(s.buffers[yi].buf,expected_y.data(),expected_y.size()) ||
         std::memcmp(s.buffers[ci].buf,expected_counts.data(),expected_counts.size()))
        throw std::runtime_error("Output changed at profile "+std::to_string(p));
    }
    return std::array<double,3>{ms(a,c),ms(a,b),ms(b,c)};
  };
  run(0);
  expected_y.assign(s.buffers[yi].buf,s.buffers[yi].buf+s.buffers[yi].size);
  expected_counts.assign(s.buffers[ci].buf,s.buffers[ci].buf+s.buffers[ci].size);
  for(int p=0;p<4;p++) {
    run(p); std::string key=std::to_string(profiles[p][0])+"_"+std::to_string(profiles[p][1]);
    write(outdir+"/"+key+"_y.bin",s.buffers[yi]); write(outdir+"/"+key+"_counts.bin",s.buffers[ci]);
    resource_snapshot(key);
  }
  std::ofstream csv(outdir+"/samples.csv"); csv<<std::setprecision(9)<<"round,mode,profile,total_ms,set_ms,exec_ms\n";
  auto sample=[&](int round,const char *mode,int p) {
    auto ts=run(p); csv<<round<<","<<mode<<","<<profiles[p][0]<<"_"<<profiles[p][1];
    for(auto t:ts) csv<<","<<t;
    csv<<"\n";
  };
  for(int round=0;round<3;round++) {
    auto solo=[&]() { for(int j=0;j<4;j++) { int p=(round%2)?3-j:j;
      for(int w=0;w<10;w++) run(p);
      for(int i=0;i<N;i++) sample(round,"solo",p);
    }};
    auto cycle=[&]() { for(int w=0;w<40;w++) run(w%4);
      for(int i=0;i<4*N;i++) sample(round,"cycle",(round%2)?3-i%4:i%4);
    };
    if(round%2) { cycle(); solo(); } else { solo(); cycle(); }
  }
  csv.close(); if(!csv) throw std::runtime_error("Could not write samples");
  std::memcpy(s.buffers[xi].buf,stress.data(),stress.size());
  run(0,false); write(outdir+"/stress_full_y.bin",s.buffers[yi]);
  run(3,false); write(outdir+"/stress_y.bin",s.buffers[yi]); write(outdir+"/stress_counts.bin",s.buffers[ci]);
  // Probe an unlisted shape last; bind success need not imply exact specialization.
  setshape(48,48); QStatus unsupported=qaicExecObjSetDataExt(s.exec,nb,s.buffers.data(),dims.data());
  QStatus unlisted_enqueue=QS_ERROR,unlisted_wait=QS_ERROR;
  if(gi>=0) {
    for(int i=0;i<48;i++) for(int j=0;j<48;j++) reinterpret_cast<int32_t*>(s.buffers[gi].buf)[i*48+j]=i+j;
  }
  // Rebind after filling the host buffer, just as in normal execution.
  if(unsupported==QS_SUCCESS) {
    CK(qaicExecObjSetDataExt(s.exec,nb,s.buffers.data(),dims.data()));
    unlisted_enqueue=qaicEnqueueExecObj(s.queue,s.exec,nullptr);
    if(unlisted_enqueue==QS_SUCCESS) unlisted_wait=qaicExecObjWaitForCompletion(s.exec,60000000);
    if(unlisted_wait==QS_SUCCESS) write(outdir+"/unlisted_y.bin",s.buffers[yi]);
  }
  std::ofstream meta(outdir+"/metadata.json");
  meta<<"{\"num_programs\":"<<qi->numPrograms<<",\"num_constants\":"<<qi->numConstants
      <<",\"constants_bytes\":"<<constants_bytes<<",\"program_req_mem_bytes\":"<<qi->programInfo[0].reqMem
      <<",\"program_loads\":1,\"program_activations\":1,\"exec_objects\":1,\"devices\":\""<<deviceids
      <<"\",\"unlisted_shape_setdata_status\":"<<int(unsupported)
      <<",\"unlisted_shape_enqueue_status\":"<<int(unlisted_enqueue)
      <<",\"unlisted_shape_wait_status\":"<<int(unlisted_wait)<<"}\n";
  meta.close(); if(!meta) throw std::runtime_error("Could not write metadata");
  std::cout<<"ALL_NORMAL_OUTPUTS_EXACT; unsupported shape setData status="<<unsupported<<"\n";
  return 0;
} catch(const std::exception &e) { std::cerr<<"ERROR "<<e.what()<<"\n"; return 1; }
