// One resident program: eight capacity shapes plus runtime expert-ID values.
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
static double ms(Clock::time_point a,Clock::time_point b) {return std::chrono::duration<double,std::milli>(b-a).count();}
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
struct Profile {std::string name;uint32_t tag=0;std::array<uint32_t,8> cap{};};

int main(int argc,char **argv) try {
  if(argc!=8)throw std::runtime_error("usage: host QPC_DIR INPUT_DIR OUT_DIR DEVICE H ITERATIONS multi|single32");
  std::string qpcdir=argv[1],root=argv[2],out=argv[3],variant=argv[7];
  QID device=std::stoi(argv[4]);uint32_t H=std::stoul(argv[5]);int N=std::stoi(argv[6]);
  if((variant!="multi"&&variant!="single32")||N<1||H<128||std::filesystem::exists(out))throw std::runtime_error("Invalid arguments");
  std::vector<Profile> profiles;std::ifstream pf(root+"/profiles.txt");Profile next;
  while(pf>>next.name>>next.tag) {
    for(auto &c:next.cap)if(!(pf>>c)||c<1||c>128)throw std::runtime_error("Invalid profile capacity");
    profiles.push_back(next);
  }
  if(profiles.size()!=5||profiles[0].name!="full128"||profiles[1].name!="uniform32"||profiles[2].name!="uniform16")
    throw std::runtime_error("Unexpected profile manifest");
  for(size_t i=0;i<profiles.size();i++)if(profiles[i].tag!=i+1)throw std::runtime_error("Invalid profile tag");
  std::vector<int> active=variant=="multi"?std::vector<int>{0,1,2,3,4}:std::vector<int>{1};
  auto allowed=[&](const Profile &p) {for(auto i:active)if(p.tag==profiles[i].tag&&p.cap==profiles[i].cap)return true;return false;};
  Profile bad=profiles[active[0]];bad.cap.fill(48);if(allowed(bad))throw std::runtime_error("Invalid shape admitted");
  bad=profiles[active[0]];bad.tag=99;if(allowed(bad))throw std::runtime_error("Invalid tag admitted");
  std::array<std::string,4> cases={"identity","swap","shuffle","duplicate_control"};
  std::array<std::array<int32_t,32>,4> ids{};
  auto legal=[](std::array<int32_t,32> v) {std::sort(v.begin(),v.end());for(int i=0;i<32;i++)if(v[i]!=i)return false;return true;};
  for(int c=0;c<4;c++) {
    auto bytes=read(root+"/orders/"+cases[c]+".bin");if(bytes.size()!=128)throw std::runtime_error("Invalid IDs size");
    std::memcpy(ids[c].data(),bytes.data(),128);
    for(auto id:ids[c])if(id<0||id>=32)throw std::runtime_error("Expert ID out of bounds");
    if(c<3&&!legal(ids[c]))throw std::runtime_error("Expected a permutation");
  }
  auto invalid_ids=ids[0];invalid_ids[0]=32;if(legal(invalid_ids))throw std::runtime_error("Invalid ID admitted");
  auto x=read(root+"/x.bin"),stress=read(root+"/stress_x.bin");
  if(x.size()!=128*H*2||stress.size()!=x.size())throw std::runtime_error("Activation size mismatch");
  std::filesystem::create_directories(out);
  std::ofstream resources(out+"/resources.csv");resources<<"stage,dram_free_kib,nsp_free\n";
  auto snapshot=[&](const std::string &name) {QResourceInfo ri{};CK(qaicGetResourceInfo(device,&ri));
    resources<<name<<","<<ri.dramFree<<","<<unsigned(ri.nspFree)<<"\n";resources.flush();};
  snapshot("before_load");Session s;
  CK(qaicCreateContext(&s.ctx,nullptr,1,&device,nullptr,nullptr,nullptr,nullptr));
  CK(qaicCreateQueue(s.ctx,&s.queue,nullptr,device));CK(qaicOpenQpcFile(&s.qpc,(qpcdir+"/programqpc.bin").c_str()));
  const QAicQpcInfo_t *qi=nullptr;CK(qaicQpcGetInfo(s.qpc,&qi));
  if(qi->numPrograms!=1)throw std::runtime_error("Expected one program");
  uint64_t constants=0;for(uint32_t i=0;i<qi->numConstants;i++)constants+=qi->constantsInfo[i].size;
  QAicProgramProperties_t pp;CK(qaicProgramPropertiesInitDefault(&pp));
  CK(qaicCreateProgram(s.ctx,&s.program,&pp,device,"adaptive_moe",s.qpc));
  CK(qaicLoadProgram(s.program));s.loaded=true;
  CK(qaicRunActivationCmd(s.program,QAIC_PROGRAM_CMD_ACTIVATE_FULL));s.active=true;snapshot("activated");
  QData iod{};CK(qaicProgramGetIoDescriptor(s.program,&iod));
  QAicExecObjProperties_t ep=QAIC_EXECOBJ_PROPERTIES_DEFAULT;
  CK(qaicCreateExecObj(s.ctx,&s.exec,&ep,s.program,&iod,nullptr,nullptr));
  QAicIoBufferInfo_t *bi=nullptr;CK(qaicProgramGetIoBufferInfo(s.program,&bi));
  s.buffers.resize(bi->numBufferMappings);std::vector<QBufferDimensions> dims(s.buffers.size());
  std::vector<std::array<uint32_t,2>> shapes(s.buffers.size());std::vector<uint64_t> maximum(s.buffers.size());
  int xi=-1,yi=-1,ci=-1,ii=-1,ti=-1;std::array<int,8> rows;rows.fill(-1);
  for(uint32_t k=0;k<bi->numBufferMappings;k++) {
    auto &m=bi->bufferMappings[k];auto i=m.index;if(i>=s.buffers.size())throw std::runtime_error("Invalid buffer index");
    auto &b=s.buffers[i];b.size=m.size;maximum[i]=m.size;b.type=QBUFFER_TYPE_HEAP;
    b.buf=static_cast<uint8_t*>(aligned_alloc(4096,(m.size+4095)/4096*4096));if(!b.buf)throw std::bad_alloc();
    std::memset(b.buf,0,b.size);dims[i].dims=shapes[i].data();std::string name=m.bufferName;
    if(name=="x"||name=="y") {
      if(b.size!=x.size())throw std::runtime_error("Hidden buffer size mismatch");
      shapes[i]={128,H};dims[i].sizeOfElem=2;dims[i].count=2;
      if(name=="x"){xi=i;std::memcpy(b.buf,x.data(),x.size());}else yi=i;
    } else {
      dims[i].sizeOfElem=4;dims[i].count=1;shapes[i]={uint32_t(b.size/4),0};
      if(name=="expert_ids"||name=="counts") {
        if(b.size!=128)throw std::runtime_error("IDs/counts size mismatch");
        if(name=="expert_ids")ii=i;else ci=i;
      } else if(name=="tag")ti=i;
      else if(name.size()==5&&name.substr(0,4)=="rows"&&name[4]>='0'&&name[4]<='7') {
        rows[name[4]-'0']=i;for(uint32_t j=0;j<b.size/4;j++)reinterpret_cast<int32_t*>(b.buf)[j]=j;
      } else throw std::runtime_error("Unexpected input "+name);
    }
    std::cout<<name<<" max_bytes="<<m.size<<"\n";
  }
  if(xi<0||yi<0||ci<0||ii<0||ti<0||std::find(rows.begin(),rows.end(),-1)!=rows.end())throw std::runtime_error("Incomplete interface");
  std::vector<std::array<std::vector<uint8_t>,4>> expected_y(profiles.size()),expected_counts(profiles.size());
  auto overflow=[&](int p,int c) {
    const auto *counts=reinterpret_cast<const int32_t*>(s.buffers[ci].buf);int dropped=0;
    for(int j=0;j<32;j++) {
      int n=counts[ids[c][j]];if(n<0||n>128)throw std::runtime_error("Invalid count");
      dropped+=std::max(0,n-int(profiles[p].cap[j/4]));
    }
    return dropped;
  };
  auto run=[&](int p,int c,bool verify=true) {
    auto a=Clock::now();
    if(p<0||size_t(p)>=profiles.size()||c<0||c>3||!allowed(profiles[p]))throw std::runtime_error("Unlisted profile/case");
    for(int g=0;g<8;g++) {
      auto i=rows[g];uint32_t count=profiles[p].cap[g];
      if(count*4>maximum[i])throw std::runtime_error("Capacity exceeds buffer allocation");
      shapes[i][0]=count;s.buffers[i].size=count*4;
    }
    if(profiles[p].tag*4>maximum[ti])throw std::runtime_error("Tag exceeds allocation");
    shapes[ti][0]=profiles[p].tag;s.buffers[ti].size=profiles[p].tag*4;
    std::memcpy(s.buffers[ii].buf,ids[c].data(),128);
    CK(qaicExecObjSetDataExt(s.exec,s.buffers.size(),s.buffers.data(),dims.data()));auto b=Clock::now();
    CK(qaicEnqueueExecObj(s.queue,s.exec,nullptr));CK(qaicExecObjWaitForCompletion(s.exec,60000000));auto end=Clock::now();
    if(verify) {
      if(overflow(p,c))throw std::runtime_error("Unexpected normal-input overflow");
      if(!expected_y[p][c].empty()&&
         (std::memcmp(s.buffers[yi].buf,expected_y[p][c].data(),expected_y[p][c].size())||
          std::memcmp(s.buffers[ci].buf,expected_counts[p][c].data(),expected_counts[p][c].size())))
        throw std::runtime_error("Repeated output changed");
    }
    return std::array<double,3>{ms(a,end),ms(a,b),ms(b,end)};
  };
  for(auto p:active)for(int c=0;c<4;c++) {
    run(p,c);expected_y[p][c].assign(s.buffers[yi].buf,s.buffers[yi].buf+s.buffers[yi].size);
    expected_counts[p][c].assign(s.buffers[ci].buf,s.buffers[ci].buf+s.buffers[ci].size);
    auto key=profiles[p].name+"_"+cases[c];write(out+"/"+key+"_y.bin",s.buffers[yi]);write(out+"/"+key+"_counts.bin",s.buffers[ci]);
    snapshot(key);
  }
  std::ofstream csv(out+"/samples.csv");csv<<std::setprecision(9)<<"round,mode,profile,case,total_ms,set_ms,exec_ms\n";
  auto sample=[&](int round,const char *mode,int p,int c) {
    auto ts=run(p,c);csv<<round<<","<<mode<<","<<profiles[p].name<<","<<cases[c];
    for(auto t:ts)csv<<","<<t;
    csv<<"\n";
  };
  for(int round=0;round<3;round++) {
    auto order=active;if(round%2)std::reverse(order.begin(),order.end());
    auto held=[&]() {for(auto p:order)for(int c=0;c<3;c++) {
      for(int w=0;w<5;w++)run(p,c);
      for(int n=0;n<N;n++)sample(round,"held",p,c);
    }};
    auto both=[&]() {
      // 5 profiles and 3 cases are coprime: this visits their full cross-product.
      for(int w=0;w<30;w++)run(order[w%order.size()],w%3);
      for(int n=0;n<N*int(order.size())*3;n++)sample(round,"both",order[n%order.size()],n%3);
    };
    if(round%2){both();held();}else{held();both();}
    for(auto p:order) {
      for(int w=0;w<15;w++)run(p,w%3);
      for(int n=0;n<N*3;n++)sample(round,"ids_only",p,n%3);
    }
    for(int c=0;c<3;c++) {
      for(int w=0;w<15;w++)run(order[w%order.size()],c);
      for(int n=0;n<N*int(order.size());n++)sample(round,"shape_only",order[n%order.size()],c);
    }
  }
  csv.close();if(!csv)throw std::runtime_error("Failed writing timing samples");
  if(variant=="multi") {
    std::memcpy(s.buffers[xi].buf,stress.data(),stress.size());
    std::ofstream recovery(out+"/recovery.csv");
    recovery<<std::setprecision(9)<<"case,profile,iteration,overflow,truncated_diff,recovered_exact,low_ms,replay_ms,total_ms\n";
    for(int c=0;c<3;c++) {
      run(0,c,false);if(overflow(0,c))throw std::runtime_error("Conservative profile overflow");
      std::vector<uint8_t> full(s.buffers[yi].buf,s.buffers[yi].buf+s.buffers[yi].size);
      write(out+"/stress_"+cases[c]+"_full_y.bin",s.buffers[yi]);write(out+"/stress_"+cases[c]+"_counts.bin",s.buffers[ci]);
      for(int p:{2,4}) {
        // Save the intentionally truncated diagnostic outside recovery timing.
        run(p,c,false);bool differs=std::memcmp(full.data(),s.buffers[yi].buf,full.size())!=0;
        if(!differs)throw std::runtime_error("Stress did not expose truncation");
        write(out+"/stress_"+cases[c]+"_"+profiles[p].name+"_truncated_y.bin",s.buffers[yi]);
        for(int n=0;n<20;n++) {
          auto start=Clock::now();auto low=run(p,c,false);int dropped=overflow(p,c);
          if(dropped<=0)throw std::runtime_error("Stress overflow not detected");
          // Reject the truncated output and replay the same input/IDs in the same program.
          auto replay=run(0,c,false);auto end=Clock::now();
          bool exact=std::memcmp(full.data(),s.buffers[yi].buf,full.size())==0;
          if(!exact||overflow(0,c))throw std::runtime_error("Recovery differs from direct full-capacity output");
          recovery<<cases[c]<<","<<profiles[p].name<<","<<n<<","<<dropped<<","<<differs<<","<<exact
                  <<","<<low[0]<<","<<replay[0]<<","<<ms(start,end)<<"\n";
        }
        write(out+"/stress_"+cases[c]+"_"+profiles[p].name+"_recovered_y.bin",s.buffers[yi]);
      }
    }
    recovery.close();if(!recovery)throw std::runtime_error("Failed writing recovery results");
    // Verify recovery leaves the program usable with the normal activation input.
    std::memcpy(s.buffers[xi].buf,x.data(),x.size());run(1,2);snapshot("after_recovery");
  }
  std::ofstream meta(out+"/metadata.json");
  meta<<"{\"program_loads\":1,\"program_activations\":1,\"exec_objects\":1,\"num_programs\":"<<qi->numPrograms
      <<",\"constants_bytes\":"<<constants<<",\"program_req_mem_bytes\":"<<qi->programInfo[0].reqMem
      <<",\"device\":"<<device<<",\"invalid_profiles_rejected_before_enqueue\":2,\"invalid_id_rejected_before_enqueue\":true}\n";
  meta.close();if(!meta)throw std::runtime_error("Failed writing metadata");
  std::cout<<"ALL_REPETITIONS_EXACT\n";
  if(variant=="multi")std::cout<<"RECOVERY_CHECKS_PASSED\n";
  return 0;
} catch(const std::exception &e) {std::cerr<<"ERROR "<<e.what()<<"\n";return 1;}
