#pragma once
#include "QAicApi.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
using Clock=std::chrono::steady_clock;
inline double ms(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
inline void check(QStatus s,const char *name){if(s!=QS_SUCCESS)throw std::runtime_error(std::string(name)+": "+std::to_string(s));}
#define CK(x) check(x,#x)
inline std::vector<uint8_t> readbin(const std::string &p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("Cannot read "+p);return {(std::istreambuf_iterator<char>(f)),{}};}
inline void writebin(const std::string &p,const QBuffer &b){std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(b.buf),b.size);if(!f)throw std::runtime_error("Cannot write "+p);}
inline bool equal(const QBuffer &a,const std::vector<uint8_t> &b){return a.size==b.size()&&!std::memcmp(a.buf,b.data(),a.size);}
struct Profile{std::string name;uint32_t tokens=0,tag=0;std::vector<uint32_t> widths,caps,limits;uint32_t work=0;};
inline std::vector<Profile> profiles(const std::string &root,uint32_t experts){
  std::ifstream f(root+"/profiles.txt");if(!f)throw std::runtime_error("Missing profiles");
  std::vector<Profile> result;Profile p;uint32_t groups;
  while(f>>p.name>>p.tokens>>p.tag>>groups){
    p.widths.clear();p.caps.clear();p.limits.clear();p.work=0;
    for(uint32_t g=0;g<groups;g++){uint32_t w,c;if(!(f>>w>>c)||!w||!c||c>p.tokens)throw std::runtime_error("Invalid group");p.widths.push_back(w);p.caps.push_back(c);p.limits.insert(p.limits.end(),w,c);p.work+=w*c;}
    if(p.limits.size()!=experts)throw std::runtime_error("Group coverage mismatch");
    result.push_back(p);
  }
  if(result.empty())throw std::runtime_error("Empty profile list");
  return result;
}
inline std::vector<int32_t> order(const int32_t *counts,uint32_t experts,bool sorted){
  std::vector<int32_t> ids(experts);std::iota(ids.begin(),ids.end(),0);
  if(sorted)std::stable_sort(ids.begin(),ids.end(),[&](int a,int b){return counts[a]>counts[b];});
  return ids;
}
inline bool safe(const Profile &p,const int32_t *counts,const std::vector<int32_t> &ids,uint32_t tokens){
  if(p.tokens!=tokens||p.limits.size()!=ids.size())return false;
  std::vector<bool> seen(ids.size());
  for(size_t j=0;j<ids.size();j++){
    auto e=ids[j];if(e<0||size_t(e)>=ids.size()||seen[e]||counts[e]<0||counts[e]>int(tokens)||counts[e]>int(p.limits[j]))return false;
    seen[e]=true;
  }
  return true;
}
inline size_t choose(const std::vector<Profile> &ps,const int32_t *counts,const std::vector<int32_t> &ids,uint32_t tokens){
  size_t best=ps.size();uint32_t work=UINT32_MAX;
  for(size_t i=0;i<ps.size();i++)if(safe(ps[i],counts,ids,tokens)&&ps[i].work<work){best=i;work=ps[i].work;}
  if(best==ps.size())throw std::runtime_error("No safe compiled profile");
  return best;
}
struct Program{
  QAicQpcObj *qpc=nullptr;QAicProgram *program=nullptr;QAicExecObj *exec=nullptr;QAicQueue *queue=nullptr;QID device=0;
  bool loaded=false,active=false;uint32_t experts=0,layers=1;
  std::vector<QBuffer> buffers;std::vector<QBufferDimensions> dims;std::vector<std::vector<uint32_t>> shapes;
  std::vector<uint64_t> maximum;std::map<std::string,size_t> index;
  QBuffer &at(const std::string &name){return buffers.at(index.at(name));}
  void copy(const std::string &name,const void *data,size_t size){auto &b=at(name);if(b.size!=size)throw std::runtime_error("Copy mismatch "+name);std::memcpy(b.buf,data,size);}
  void shape(uint32_t tokens,uint32_t tag=0){
    for(auto &entry:index){auto name=entry.first;auto i=entry.second;
      if(name=="x"||name=="y"||name=="route_weights")shapes[i][0]=tokens;
      else if(name=="profile_tag")shapes[i][0]=tag;
      else if(name=="layout_tag")shapes[i][0]=(tag-1)%6+1;
      uint64_t bytes=dims[i].sizeOfElem;for(auto d:shapes[i])bytes*=d;
      if(!bytes||bytes>maximum[i])throw std::runtime_error("Invalid shape "+name);
      buffers[i].size=bytes;
    }
  }
  void bind(){CK(qaicExecObjSetDataExt(exec,buffers.size(),buffers.data(),dims.data()));}
  void enqueue(){CK(qaicEnqueueExecObj(queue,exec,nullptr));}
  void wait(){CK(qaicExecObjWaitForCompletion(exec,60000000));}
  void run(){bind();enqueue();wait();}
};
struct Session{
  QAicContext *ctx=nullptr;std::map<QID,QAicQueue*> queues;std::vector<std::unique_ptr<Program>> programs;
  std::function<void(const std::string&,QID)> snapshot;
  ~Session(){
    for(auto &p:programs){if(p->exec)qaicReleaseExecObj(p->exec);if(p->active)qaicRunActivationCmd(p->program,QAIC_PROGRAM_CMD_DEACTIVATE_FULL);if(p->loaded)qaicUnloadProgram(p->program);if(p->program)qaicReleaseProgram(p->program);if(p->qpc)qaicCloseQpc(p->qpc);for(auto &b:p->buffers)free(b.buf);}
    for(auto &q:queues)qaicReleaseQueue(q.second);
    if(ctx)qaicReleaseContext(ctx);
  }
  void init(std::vector<QID> devices){CK(qaicCreateContext(&ctx,nullptr,devices.size(),devices.data(),nullptr,nullptr,nullptr,nullptr));for(auto d:devices){QAicQueue *q=nullptr;CK(qaicCreateQueue(ctx,&q,nullptr,d));queues[d]=q;}}
  Program &add(const std::string &path,QID device,uint32_t experts,uint32_t layers,std::ostream &meta){
    programs.push_back(std::make_unique<Program>());auto &p=*programs.back();p.device=device;p.experts=experts;p.layers=layers;p.queue=queues.at(device);
    CK(qaicOpenQpcFile(&p.qpc,(path+"/programqpc.bin").c_str()));
    const QAicQpcInfo_t *info=nullptr;CK(qaicQpcGetInfo(p.qpc,&info));
    if(info->numPrograms!=1)throw std::runtime_error("Expected one program");
    // SDK sizes are uint32 fields; audit extracted constants.bin for true storage above 4 GiB.
    uint64_t constants=0;for(uint32_t i=0;i<info->numConstants;i++)constants+=info->constantsInfo[i].size;
    meta<<path<<","<<device<<","<<constants<<","<<info->programInfo[0].reqMem<<","<<info->programInfo[0].numCores<<"\n";meta.flush();
    QAicProgramProperties_t pp;CK(qaicProgramPropertiesInitDefault(&pp));CK(qaicCreateProgram(ctx,&p.program,&pp,device,"joint_moe",p.qpc));
    CK(qaicLoadProgram(p.program));p.loaded=true;if(snapshot)snapshot("loaded_"+std::to_string(programs.size()),device);
    CK(qaicRunActivationCmd(p.program,QAIC_PROGRAM_CMD_ACTIVATE_FULL));p.active=true;if(snapshot)snapshot("active_"+std::to_string(programs.size()),device);
    QData iod{};CK(qaicProgramGetIoDescriptor(p.program,&iod));QAicExecObjProperties_t ep=QAIC_EXECOBJ_PROPERTIES_DEFAULT;
    CK(qaicCreateExecObj(ctx,&p.exec,&ep,p.program,&iod,nullptr,nullptr));
    QAicIoBufferInfo_t *bi=nullptr;CK(qaicProgramGetIoBufferInfo(p.program,&bi));auto n=bi->numBufferMappings;
    p.buffers.resize(n);p.dims.resize(n);p.shapes.resize(n);p.maximum.resize(n);
    for(uint32_t k=0;k<n;k++){
      auto &m=bi->bufferMappings[k];size_t i=m.index;if(i>=n)throw std::runtime_error("Bad buffer mapping");std::string name=m.bufferName;
      if(!p.index.emplace(name,i).second)throw std::runtime_error("Duplicate buffer");
      auto &b=p.buffers[i];b.size=m.size;p.maximum[i]=m.size;b.type=QBUFFER_TYPE_HEAP;b.buf=static_cast<uint8_t*>(aligned_alloc(4096,(m.size+4095)/4096*4096));if(!b.buf)throw std::bad_alloc();std::memset(b.buf,0,b.size);
      auto &d=p.dims[i];d.sizeOfElem=(name=="x"||name=="y"||name=="route_weights")?2:4;
      if(d.sizeOfElem==2){uint32_t width=name=="route_weights"?128:2048;p.shapes[i]={uint32_t(b.size/(width*2)),width};}
      else if(name=="counts"&&layers>1)p.shapes[i]={layers,experts};
      else p.shapes[i]={uint32_t(b.size/4)};
      d.count=p.shapes[i].size();d.dims=p.shapes[i].data();
      std::cout<<"IO "<<name<<" max_bytes="<<b.size<<"\n";
    }
    return p;
  }
};
struct Workload{std::string name;uint32_t tokens=0;std::vector<uint8_t> x,routes,counts;};
inline std::vector<Workload> workloads(const std::string &root,uint32_t experts,uint32_t layers=1){
  std::ifstream f(root+"/workloads.txt");std::vector<Workload> ws;Workload w;
  while(f>>w.name>>w.tokens){w.x=readbin(root+"/inputs/"+w.name+"_x.bin");w.counts=readbin(root+"/reference/"+w.name+"_counts.bin");
    if(layers==1)w.routes=readbin(root+"/reference/"+w.name+"_route_weights.bin");
    if(w.x.size()!=w.tokens*2048*2||w.counts.size()!=experts*layers*4)throw std::runtime_error("Invalid workload sizes");
    ws.push_back(w);}
  if(ws.empty())throw std::runtime_error("Missing workloads");
  return ws;
}
