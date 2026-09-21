// Change shape-specialized topology and resident expert IDs in one program.
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
#include <stdexcept>
#include <string>
#include <vector>

using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point a,Clock::time_point b) {return std::chrono::duration<double,std::milli>(b-a).count();}
static void check(QStatus s,const char *what) {if(s!=QS_SUCCESS)throw std::runtime_error(std::string(what)+": "+std::to_string(s));}
#define CK(x) check(x,#x)
static std::vector<uint8_t> read(const std::string &path) {
  std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("Cannot read "+path);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),{});
}
static void write(const std::string &path,const QBuffer &b) {
  std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(b.buf),b.size);
  if(!f)throw std::runtime_error("Cannot write "+path);
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
  if(argc!=8)throw std::runtime_error("usage: host QPC INPUT OUT DEVICE H ITERATIONS multi|single_wN|fixed_wN");
  std::string qpcdir=argv[1],root=argv[2],out=argv[3],variant=argv[7];QID device=std::stoi(argv[4]);
  uint32_t H=std::stoul(argv[5]);int N=std::stoi(argv[6]);bool multi=variant=="multi";
  bool tagged=multi||variant.rfind("single_w",0)==0;
  const std::vector<int> all_widths={1,2,4,8,16,32};std::vector<int> widths=all_widths;
  if(!multi) {
    if(!tagged&&variant.rfind("fixed_w",0)!=0)throw std::runtime_error("Unknown variant");
    widths={std::stoi(variant.substr(variant.find('_')+2))};
    if(std::find(all_widths.begin(),all_widths.end(),widths[0])==all_widths.end())throw std::runtime_error("Unlisted width");
  }
  auto allowed=[&](int width){return std::find(widths.begin(),widths.end(),width)!=widths.end();};
  if(allowed(3)||allowed(0)||allowed(64))throw std::runtime_error("Whitelist admits invalid topology");
  if(device!=0||N<1||H<128||std::filesystem::exists(out))throw std::runtime_error("Invalid arguments");
  auto x=read(root+"/x.bin");if(x.size()!=128*H*2)throw std::runtime_error("Activation size mismatch");
  std::array<std::string,4> cases={"identity","swap","shuffle","duplicate_control"};
  std::array<std::vector<uint8_t>,4> ids;
  for(int c=0;c<4;c++) {
    ids[c]=read(root+"/orders/"+cases[c]+".bin");if(ids[c].size()!=128)throw std::runtime_error("IDs size mismatch");
    std::array<int32_t,32> order{};std::memcpy(order.data(),ids[c].data(),128);
    for(auto id:order)if(id<0||id>=32)throw std::runtime_error("ID out of bounds");
    if(c<3){std::sort(order.begin(),order.end());for(int e=0;e<32;e++)if(order[e]!=e)throw std::runtime_error("Not a permutation");}
  }
  std::filesystem::create_directories(out);std::ofstream resources(out+"/resources.csv");
  resources<<"stage,dram_free_kib,nsp_free\n";
  auto snapshot=[&](const std::string &name){QResourceInfo ri{};CK(qaicGetResourceInfo(device,&ri));
    resources<<name<<","<<ri.dramFree<<","<<unsigned(ri.nspFree)<<"\n";resources.flush();};
  snapshot("before");Session s;auto setup_start=Clock::now();
  CK(qaicCreateContext(&s.ctx,nullptr,1,&device,nullptr,nullptr,nullptr,nullptr));
  CK(qaicCreateQueue(s.ctx,&s.queue,nullptr,device));CK(qaicOpenQpcFile(&s.qpc,(qpcdir+"/programqpc.bin").c_str()));
  const QAicQpcInfo_t *qi=nullptr;CK(qaicQpcGetInfo(s.qpc,&qi));if(qi->numPrograms!=1)throw std::runtime_error("Expected one program");
  uint64_t constants=0;for(uint32_t i=0;i<qi->numConstants;i++)constants+=qi->constantsInfo[i].size;
  QAicProgramProperties_t pp;CK(qaicProgramPropertiesInitDefault(&pp));
  CK(qaicCreateProgram(s.ctx,&s.program,&pp,device,"topology_probe",s.qpc));
  CK(qaicLoadProgram(s.program));s.loaded=true;
  CK(qaicRunActivationCmd(s.program,QAIC_PROGRAM_CMD_ACTIVATE_FULL));s.active=true;snapshot("activated");
  QData iod{};CK(qaicProgramGetIoDescriptor(s.program,&iod));QAicExecObjProperties_t ep=QAIC_EXECOBJ_PROPERTIES_DEFAULT;
  CK(qaicCreateExecObj(s.ctx,&s.exec,&ep,s.program,&iod,nullptr,nullptr));
  QAicIoBufferInfo_t *bi=nullptr;CK(qaicProgramGetIoBufferInfo(s.program,&bi));
  s.buffers.resize(bi->numBufferMappings);std::vector<QBufferDimensions> dims(s.buffers.size());
  std::vector<std::array<uint32_t,2>> shapes(s.buffers.size());int xi=-1,yi=-1,ci=-1,ii=-1,ti=-1;
  uint64_t tag_max=0;
  for(uint32_t k=0;k<bi->numBufferMappings;k++) {
    auto &m=bi->bufferMappings[k];size_t i=m.index;if(i>=s.buffers.size())throw std::runtime_error("Invalid mapping");
    auto &b=s.buffers[i];b.size=m.size;b.type=QBUFFER_TYPE_HEAP;
    b.buf=static_cast<uint8_t*>(aligned_alloc(4096,(m.size+4095)/4096*4096));if(!b.buf)throw std::bad_alloc();
    std::memset(b.buf,0,b.size);dims[i].dims=shapes[i].data();std::string name=m.bufferName;
    if(name=="x"||name=="y") {
      shapes[i]={128,H};dims[i].sizeOfElem=2;dims[i].count=2;if(b.size!=x.size())throw std::runtime_error("Hidden size mismatch");
      if(name=="x"){xi=i;std::memcpy(b.buf,x.data(),x.size());}else yi=i;
    } else {
      shapes[i]={uint32_t(b.size/4),0};dims[i].sizeOfElem=4;dims[i].count=1;
      if(name=="expert_ids"||name=="counts") {
        if(b.size!=128)throw std::runtime_error("IDs/counts size mismatch");
        if(name=="expert_ids")ii=i;else ci=i;
      } else if(name=="layout_tag"){ti=i;tag_max=b.size;}else throw std::runtime_error("Unknown IO "+name);
    }
    std::cout<<name<<" max_bytes="<<b.size<<"\n";
  }
  if(xi<0||yi<0||ci<0||ii<0||tagged!=(ti>=0))throw std::runtime_error("Interface mismatch");
  double setup_ms=ms(setup_start,Clock::now());
  std::map<int,std::array<std::vector<uint8_t>,4>> expected,expected_counts;
  auto run=[&](int width,int c) {
    auto start=Clock::now();if(!allowed(width)||c<0||c>3)throw std::runtime_error("Unlisted width/case");
    std::memcpy(s.buffers[ii].buf,ids[c].data(),128);
    if(tagged) {
      if(uint64_t(width)*4>tag_max)throw std::runtime_error("Tag exceeds allocation");
      shapes[ti][0]=width;s.buffers[ti].size=width*4;
    }
    CK(qaicExecObjSetDataExt(s.exec,s.buffers.size(),s.buffers.data(),dims.data()));auto bound=Clock::now();
    CK(qaicEnqueueExecObj(s.queue,s.exec,nullptr));CK(qaicExecObjWaitForCompletion(s.exec,60000000));auto end=Clock::now();
    const auto *counts=reinterpret_cast<const int32_t*>(s.buffers[ci].buf);
    for(int e=0;e<32;e++)if(counts[e]<0||counts[e]>32)throw std::runtime_error("Unexpected overflow/count");
    if(!expected[width][c].empty()&&
       (std::memcmp(s.buffers[yi].buf,expected[width][c].data(),s.buffers[yi].size)||
        std::memcmp(s.buffers[ci].buf,expected_counts[width][c].data(),128)))throw std::runtime_error("Repeated output changed");
    return std::array<double,3>{ms(start,end),ms(start,bound),ms(bound,end)};
  };
  for(auto width:widths)for(int c=0;c<4;c++) {
    run(width,c);expected[width][c].assign(s.buffers[yi].buf,s.buffers[yi].buf+s.buffers[yi].size);
    expected_counts[width][c].assign(s.buffers[ci].buf,s.buffers[ci].buf+128);
    std::string key="w"+std::to_string(width)+"_"+cases[c];
    write(out+"/"+key+"_y.bin",s.buffers[yi]);write(out+"/"+key+"_counts.bin",s.buffers[ci]);snapshot(key);
  }
  std::ofstream csv(out+"/samples.csv");csv<<std::setprecision(9)<<"round,mode,width,case,total_ms,bind_ms,execute_ms\n";
  auto sample=[&](int round,const char *mode,int width,int c){auto t=run(width,c);
    csv<<round<<","<<mode<<","<<width<<","<<cases[c];for(auto v:t)csv<<","<<v;csv<<"\n";};
  for(int round=0;round<3;round++) {
    auto order=widths;if(round%2)std::reverse(order.begin(),order.end());
    auto held=[&](){for(auto width:order)for(int c=0;c<3;c++) {
      for(int w=0;w<5;w++)run(width,c);
      for(int n=0;n<N;n++)sample(round,"held",width,c);
    }};
    auto both=[&](){int size=order.size();
      for(int w=0;w<size*6;w++)run(order[w%size],(w/size+w)%3);
      for(int n=0;n<N*size*3;n++)sample(round,"both",order[n%size],(n/size+n)%3);
    };
    if(multi&&round%2){both();held();}else {held();if(multi)both();}
    for(auto width:order) {
      for(int w=0;w<15;w++)run(width,w%3);
      for(int n=0;n<N*3;n++)sample(round,"ids_only",width,n%3);
    }
    if(multi)for(int c=0;c<3;c++) {
      for(int w=0;w<int(order.size())*3;w++)run(order[w%order.size()],c);
      for(int n=0;n<N*int(order.size());n++)sample(round,"shape_only",order[n%order.size()],c);
    }
    snapshot("round"+std::to_string(round));
  }
  csv.close();if(!csv)throw std::runtime_error("CSV write failed");
  std::ofstream meta(out+"/metadata.json");meta<<"{\"variant\":\""<<variant<<"\",\"program_loads\":1,\"program_activations\":1,\"exec_objects\":1,"
    <<"\"device\":0,\"setup_ms\":"<<setup_ms<<",\"constant_segments_bytes\":"<<constants
    <<",\"iterations\":"<<N<<",\"rounds\":3,\"invalid_widths_rejected_before_enqueue\":true}\n";
  meta.close();if(!meta)throw std::runtime_error("Metadata write failed");
  std::cout<<"TOPOLOGY_SWITCH_CHECKS_PASSED "<<variant<<"\n";return 0;
} catch(const std::exception &e) {std::cerr<<e.what()<<"\n";return 1;}
