// Current on-card routing -> host policy -> shape-specialized resident experts.
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
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point a,Clock::time_point b) {return std::chrono::duration<double,std::milli>(b-a).count();}
static void check(QStatus s,const char *name) {if(s!=QS_SUCCESS)throw std::runtime_error(std::string(name)+": "+std::to_string(s));}
#define CK(x) check(x,#x)
static std::vector<uint8_t> read(const std::string &p) {
  std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("Cannot read "+p);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),{});
}
static void write(const std::string &p,const QBuffer &b) {
  std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(b.buf),b.size);
  if(!f)throw std::runtime_error("Cannot write "+p);
}
struct Profile {std::string name;uint32_t tag=0;std::array<uint32_t,8> cap{};};
struct Choice {int profile=0;std::array<int32_t,32> ids{};};
static Choice choose(const int32_t *counts,const std::vector<Profile> &profiles) {
  Choice choice;std::iota(choice.ids.begin(),choice.ids.end(),0);
  for(int e=0;e<32;e++)if(counts[e]<0||counts[e]>128)throw std::runtime_error("Count out of range");
  std::stable_sort(choice.ids.begin(),choice.ids.end(),[&](int a,int b){return counts[a]>counts[b];});
  uint32_t best=UINT32_MAX;bool found=false;
  for(size_t p=0;p<profiles.size();p++) {
    bool safe=true;for(int j=0;j<32;j++)safe &= counts[choice.ids[j]]<=int(profiles[p].cap[j/4]);
    uint32_t work=4*std::accumulate(profiles[p].cap.begin(),profiles[p].cap.end(),0u);
    if(safe&&work<best){choice.profile=p;best=work;found=true;}
  }
  if(!found)throw std::runtime_error("No safe compiled profile");
  return choice;
}
struct Program {
  QAicQpcObj *qpc=nullptr;QAicProgram *program=nullptr;QAicExecObj *exec=nullptr;
  bool loaded=false,active=false;
  std::vector<QBuffer> buffers;
  std::vector<QBufferDimensions> dims;
  std::vector<std::array<uint32_t,2>> shapes;
  std::vector<uint64_t> maximum;
  std::map<std::string,size_t> index;
  QBuffer &at(const std::string &name) {return buffers.at(index.at(name));}
  void copy(const std::string &name,const void *data,size_t size) {
    auto &b=at(name);if(b.size!=size)throw std::runtime_error("Copy size mismatch: "+name);
    std::memcpy(b.buf,data,size);
  }
  void bind(const Profile &p,const std::array<int32_t,32> &ids) {
    copy("expert_ids",ids.data(),128);
    for(int g=0;g<9;g++) {
      std::string name=g<8?"rows"+std::to_string(g):"tag";
      uint32_t n=g<8?p.cap[g]:p.tag;auto i=index.at(name);
      if(n*4>maximum[i])throw std::runtime_error("Shape exceeds allocation");
      shapes[i][0]=n;buffers[i].size=n*4;
    }
  }
  void run(QAicQueue *q) {
    CK(qaicExecObjSetDataExt(exec,buffers.size(),buffers.data(),dims.data()));
    CK(qaicEnqueueExecObj(q,exec,nullptr));CK(qaicExecObjWaitForCompletion(exec,60000000));
  }
};
struct Session {
  QAicContext *ctx=nullptr;QAicQueue *queue=nullptr;QAicProgramGroup *group=nullptr;bool enabled=false;
  std::vector<std::unique_ptr<Program>> programs;
  ~Session() {
    for(auto &p:programs)if(p->exec)qaicReleaseExecObj(p->exec);
    if(group) {
      if(enabled)qaicProgramGroupDisable(group,true,60000000);
      qaicReleaseProgramGroup(group);
    } else for(auto &p:programs) {
      if(p->active)qaicRunActivationCmd(p->program,QAIC_PROGRAM_CMD_DEACTIVATE_FULL);
      if(p->loaded)qaicUnloadProgram(p->program);
      if(p->program)qaicReleaseProgram(p->program);
    }
    for(auto &p:programs) {
      if(p->qpc)qaicCloseQpc(p->qpc);
      for(auto &b:p->buffers)free(b.buf);
    }
    if(queue&&!group)qaicReleaseQueue(queue);
    if(ctx)qaicReleaseContext(ctx);
  }
  Program &add(const std::string &path,QID device) {
    programs.push_back(std::make_unique<Program>());auto &p=*programs.back();
    CK(qaicOpenQpcFile(&p.qpc,(path+"/programqpc.bin").c_str()));
    QAicProgramProperties_t pp;CK(qaicProgramPropertiesInitDefault(&pp));
    if(group)CK(qaicProgramGroupAddProgram(group,p.qpc,&p.program,&pp));
    else {
      CK(qaicCreateProgram(ctx,&p.program,&pp,device,"current_count",p.qpc));
      CK(qaicLoadProgram(p.program));p.loaded=true;
      CK(qaicRunActivationCmd(p.program,QAIC_PROGRAM_CMD_ACTIVATE_FULL));p.active=true;
    }
    return p;
  }
  void allocate(Program &p,uint32_t H) {
    QData iod{};CK(qaicProgramGetIoDescriptor(p.program,&iod));
    QAicExecObjProperties_t ep=QAIC_EXECOBJ_PROPERTIES_DEFAULT;
    CK(qaicCreateExecObj(ctx,&p.exec,&ep,p.program,&iod,nullptr,nullptr));
    QAicIoBufferInfo_t *bi=nullptr;CK(qaicProgramGetIoBufferInfo(p.program,&bi));
    auto n=bi->numBufferMappings;p.buffers.resize(n);p.dims.resize(n);p.shapes.resize(n);p.maximum.resize(n);
    for(uint32_t k=0;k<n;k++) {
      auto &m=bi->bufferMappings[k];size_t i=m.index;if(i>=n)throw std::runtime_error("Invalid mapping index");
      std::string name=m.bufferName;if(!p.index.emplace(name,i).second)throw std::runtime_error("Duplicate mapping name");
      auto &b=p.buffers[i];b.size=m.size;p.maximum[i]=m.size;b.type=QBUFFER_TYPE_HEAP;
      b.buf=static_cast<uint8_t*>(aligned_alloc(4096,(m.size+4095)/4096*4096));if(!b.buf)throw std::bad_alloc();
      std::memset(b.buf,0,b.size);auto &d=p.dims[i];d.dims=p.shapes[i].data();
      if(name=="x"||name=="y"||name=="route_weights") {
        p.shapes[i]={128,name=="route_weights"?128:H};d.sizeOfElem=2;d.count=2;
        if(b.size!=uint64_t(p.shapes[i][0])*p.shapes[i][1]*2)throw std::runtime_error("Hidden/routing size mismatch");
      } else {
        d.sizeOfElem=4;d.count=1;p.shapes[i]={uint32_t(b.size/4),0};
        if(name=="expert_ids"||name=="counts") {
          if(b.size!=128)throw std::runtime_error("IDs/counts size mismatch");
        } else if(name.size()==5&&name.substr(0,4)=="rows"&&name[4]>='0'&&name[4]<='7') {
          for(uint32_t j=0;j<b.size/4;j++)reinterpret_cast<int32_t*>(b.buf)[j]=j;
        } else if(name!="tag")throw std::runtime_error("Unknown IO "+name);
      }
      std::cout<<name<<" max_bytes="<<b.size<<"\n";
    }
  }
};
struct Workload {
  std::string name;std::vector<uint8_t> x,routes,counts,ids,expected,expected_full;
  Choice oracle;
};

int main(int argc,char **argv) try {
  if(argc!=8)throw std::runtime_error("usage: host ROOT PRECISION OUT DEVICE H ITERATIONS split|pg|pgdata|fused15|fused16");
  std::string root=argv[1],precision=argv[2],out=argv[3],mode=argv[7];QID device=std::stoi(argv[4]);
  uint32_t H=std::stoul(argv[5]);int N=std::stoi(argv[6]);
  bool grouped=mode=="pg"||mode=="pgdata";bool split=mode=="split"||grouped;
  if(device!=0||N<1||(!split&&mode!="fused15"&&mode!="fused16")||std::filesystem::exists(out))throw std::runtime_error("Invalid arguments");
  std::vector<Profile> profiles;std::ifstream pf(root+"/profiles.txt");Profile p;
  while(pf>>p.name>>p.tag) {
    for(auto &c:p.cap)if(!(pf>>c)||c<1||c>128)throw std::runtime_error("Invalid profile");
    if(p.tag!=profiles.size()+1)throw std::runtime_error("Unexpected profile tag");
    profiles.push_back(p);
  }
  if(profiles.size()!=5||profiles[0].cap!=std::array<uint32_t,8>{128,128,128,128,128,128,128,128})throw std::runtime_error("Missing conservative profile");
  std::vector<Workload> workloads;std::ifstream wf(root+"/workloads.txt");std::string name;
  while(wf>>name) {
    Workload w;w.name=name;w.x=read(root+"/inputs/"+name+"_x.bin");
    w.routes=read(root+"/reference/"+name+"_route_weights.bin");w.counts=read(root+"/reference/"+name+"_counts.bin");
    w.ids=read(root+"/reference/"+name+"_ids.bin");
    if(w.x.size()!=128*H*2||w.routes.size()!=32768||w.counts.size()!=128||w.ids.size()!=128)throw std::runtime_error("Invalid workload sizes");
    // Only fused controls use precomputed counts. Split execution selects from the router output below.
    w.oracle=choose(reinterpret_cast<const int32_t*>(w.counts.data()),profiles);workloads.push_back(std::move(w));
  }
  if(workloads.size()!=4)throw std::runtime_error("Expected four workloads");
  std::filesystem::create_directories(out);std::ofstream resources(out+"/resources.csv");
  resources<<"stage,dram_free_kib,nsp_free\n";
  auto snapshot=[&](const std::string &stage){QResourceInfo ri{};CK(qaicGetResourceInfo(device,&ri));
    resources<<stage<<","<<ri.dramFree<<","<<unsigned(ri.nspFree)<<"\n";resources.flush();};
  snapshot("before");Session s;
  CK(qaicCreateContext(&s.ctx,nullptr,1,&device,nullptr,nullptr,nullptr,nullptr));
  if(grouped) {
    QAicProgramGroupProperties_t prop;CK(qaicProgramGroupPropertiesInitDefault(&prop));
    prop.enableOversubscription=QAIC_PROGRAM_GROUP_PROPERTIES_OVERSUBSCRIPTION_ENABLED;
    if(mode=="pgdata")prop.protocolSelect=QAIC_PROGRAM_GROUP_PROTOCOL_DATA_PATH;
    CK(qaicCreateProgramGroup(s.ctx,&s.group,&s.queue,&prop,"routing_boundary",device));
    std::cout<<"ProgramGroup protocol="<<prop.protocolSelect<<"\n";
  } else CK(qaicCreateQueue(s.ctx,&s.queue,nullptr,device));
  auto start=Clock::now();Program *router=nullptr;
  if(split){router=&s.add(root+"/"+precision+"/router1",device);snapshot("router_added");}
  std::string expertpath=root+"/"+precision+"/"+(split?(grouped?"expert16":"expert15"):mode);
  if(mode=="fused16") {
    // Source location is supplied separately by the driver through this symlink.
    expertpath=root+"/"+precision+"/fused16";
  }
  Program &expert=s.add(expertpath,device);snapshot("expert_added");
  if(s.group){CK(qaicProgramGroupEnable(s.group));s.enabled=true;}
  for(auto &entry:s.programs)s.allocate(*entry,H);
  double setup=ms(start,Clock::now());snapshot("ready");
  auto equal=[](const QBuffer &b,const std::vector<uint8_t> &v){return b.size==v.size()&&std::memcmp(b.buf,v.data(),v.size())==0;};
  auto execute=[&](Workload &w,bool full=false) {
    std::array<double,6> times{};Choice choice=w.oracle;auto begin=Clock::now();
    if(router) {
      router->copy("x",w.x.data(),w.x.size());router->run(s.queue);auto routed=Clock::now();
      choice=choose(reinterpret_cast<const int32_t*>(router->at("counts").buf),profiles);auto decided=Clock::now();
      expert.copy("route_weights",router->at("route_weights").buf,router->at("route_weights").size);
      expert.copy("x",w.x.data(),w.x.size());auto handed=Clock::now();
      times[1]=ms(begin,routed);times[2]=ms(routed,decided);times[3]=ms(decided,handed);
    } else {expert.copy("x",w.x.data(),w.x.size());times[3]=ms(begin,Clock::now());}
    if(full)choice.profile=0;
    auto dispatched=Clock::now();expert.bind(profiles.at(choice.profile),choice.ids);expert.run(s.queue);auto end=Clock::now();
    times[0]=ms(begin,end);times[4]=ms(dispatched,end);times[5]=profiles[choice.profile].tag;
    // Correctness checks and file writes are outside transaction timing, for every repetition.
    if(!equal(expert.at("counts"),w.counts)||std::memcmp(choice.ids.data(),w.ids.data(),128))throw std::runtime_error("Counts/IDs mismatch");
    if(router&&(!equal(router->at("counts"),w.counts)||!equal(router->at("route_weights"),w.routes)))throw std::runtime_error("Router mismatch");
    auto counts=reinterpret_cast<const int32_t*>(expert.at("counts").buf);
    for(int j=0;j<32;j++)if(counts[choice.ids[j]]>int(profiles[choice.profile].cap[j/4]))throw std::runtime_error("Overflow");
    if(!full&&!w.expected.empty()&&!equal(expert.at("y"),w.expected))throw std::runtime_error("Repeated output changed");
    if(full&&!w.expected_full.empty()&&!equal(expert.at("y"),w.expected_full))throw std::runtime_error("Repeated conservative output changed");
    if(!full&&choice.profile!=w.oracle.profile)throw std::runtime_error("Unexpected profile selected");
    return times;
  };
  for(auto &w:workloads) {
    execute(w);auto &y=expert.at("y");w.expected.assign(y.buf,y.buf+y.size);
    for(const std::string key:{"y","counts","expert_ids"})write(out+"/"+w.name+"_"+(key=="expert_ids"?"ids":key)+".bin",expert.at(key));
    if(router)write(out+"/"+w.name+"_route_weights.bin",router->at("route_weights"));
    else {execute(w,true);w.expected_full.assign(y.buf,y.buf+y.size);write(out+"/"+w.name+"_full_y.bin",expert.at("y"));}
    snapshot(w.name);
  }
  std::ofstream csv(out+"/samples.csv"),fullcsv(out+"/full_samples.csv"),solo(out+"/solo_samples.csv");
  std::string header="round,schedule,workload,total_ms,router_ms,decision_ms,handoff_ms,expert_ms,tag\n";
  csv<<std::setprecision(9)<<header;fullcsv<<std::setprecision(9)<<header;
  solo<<std::setprecision(9)<<"round,program,total_ms\n";
  auto sample=[&](int round,const char *schedule,Workload &w,bool full=false) {
    auto ts=execute(w,full);auto &stream=full?fullcsv:csv;
    stream<<round<<","<<schedule<<","<<w.name;for(auto t:ts)stream<<","<<t;stream<<"\n";
  };
  for(int round=0;round<3;round++) {
    std::array<int,4> order={0,1,2,3};if(round%2)std::reverse(order.begin(),order.end());
    auto held=[&](){for(auto i:order){for(int w=0;w<5;w++)execute(workloads[i]);for(int n=0;n<N;n++)sample(round,"held",workloads[i]);}};
    auto cycle=[&](){for(int w=0;w<12;w++)execute(workloads[order[w%4]]);for(int n=0;n<N*4;n++)sample(round,"cycle",workloads[order[n%4]]);};
    if(round%2){cycle();held();}else{held();cycle();}
    if(!router)for(auto i:order) {
      for(int w=0;w<5;w++)execute(workloads[i],true);
      for(int n=0;n<N;n++)sample(round,"held",workloads[i],true);
    }
    if(router) {
      execute(workloads[0]);
      for(auto entry:{router,&expert}) {
        for(int w=0;w<5;w++)entry->run(s.queue);
        for(int n=0;n<N;n++) {
          auto a=Clock::now();entry->run(s.queue);auto b=Clock::now();
          solo<<round<<","<<(entry==router?"router":"expert")<<","<<ms(a,b)<<"\n";
          if(!equal(entry->at("counts"),workloads[0].counts))throw std::runtime_error("Solo counts mismatch");
          if(entry==&expert&&!equal(entry->at("y"),workloads[0].expected))throw std::runtime_error("Solo output mismatch");
        }
      }
    }
    snapshot("round"+std::to_string(round));
  }
  csv.close();fullcsv.close();solo.close();if(!csv||!fullcsv||!solo)throw std::runtime_error("CSV write failed");
  std::ofstream meta(out+"/metadata.json");meta<<"{\"mode\":\""<<mode<<"\",\"device\":0,\"setup_ms\":"<<setup
    <<",\"programs\":"<<s.programs.size()<<",\"program_group\":"<<(s.group?"true":"false")
    <<",\"host_reload_during_samples\":false,\"counts_source\":\""<<(router?"current router output":"precomputed oracle control")
    <<"\",\"iterations_per_workload_schedule_round\":"<<N<<",\"rounds\":3}\n";
  meta.close();if(!meta)throw std::runtime_error("Metadata write failed");
  std::cout<<"CURRENT_COUNT_CHECKS_PASSED "<<mode<<"\n";return 0;
} catch(const std::exception &e) {std::cerr<<e.what()<<"\n";return 1;}
