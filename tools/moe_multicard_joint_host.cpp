// Four stationary 32-expert banks, one current router, independent per-card profiles.
#include "moe_joint_runtime.h"
#include <array>
#include <immintrin.h>
int main(int argc,char **argv)try{
  if(argc!=7)throw std::runtime_error("usage: host GLOBAL_ROOT SHARD_PARENT PRECISION OUT ITERATIONS SERIAL_CONTROL");
  std::string root=argv[1],shards=argv[2],precision=argv[3],out=argv[4];int N=std::stoi(argv[5]);bool serial=std::stoi(argv[6]);
  if(N<1||std::filesystem::exists(out))throw std::runtime_error("Invalid output/iterations");
  auto ps=profiles(shards+"/card0",32);auto ws=workloads(root,128);std::filesystem::create_directories(out);
  std::ofstream res(out+"/resources.csv"),meta(out+"/packages.csv"),csv(out+"/samples.csv");
  res<<"stage,device,dram_free_kib,nsp_free\n";meta<<"path,device,sdk_constants_size32_sum,required_memory_bytes,cores\n";
  csv<<std::setprecision(9)<<"round,schedule,workload,policy,tag0,tag1,tag2,tag3,total_ms,router_ms,decision_ms,handoff_ms,dispatch_wait_ms,merge_ms\n";
  Session s;s.snapshot=[&](const std::string &stage,QID dev){QResourceInfo ri{};CK(qaicGetResourceInfo(dev,&ri));res<<stage<<","<<dev<<","<<ri.dramFree<<","<<unsigned(ri.nspFree)<<"\n";res.flush();};
  for(QID d=0;d<4;d++)s.snapshot("before",d);
  s.init({0,1,2,3});auto &router=s.add(root+"/"+precision+"/router1",0,128,1,meta);
  std::array<Program*,4> experts{};
  for(QID d=0;d<4;d++)experts[d]=&s.add(shards+"/card"+std::to_string(d)+"/"+precision+"/expert15",d,32,1,meta);
  for(QID d=0;d<4;d++)s.snapshot("ready",d);
  std::map<std::string,std::vector<uint8_t>> expected;
  std::vector<uint16_t> merged(256*2048);
  auto execute=[&](Workload &w,const std::string &policy,int round,const std::string &schedule){
    auto begin=Clock::now();router.shape(w.tokens);router.copy("x",w.x.data(),w.x.size());router.run();auto routed=Clock::now();
    auto counts=reinterpret_cast<const int32_t*>(router.at("counts").buf);
    std::array<std::vector<int32_t>,4> ids;std::array<size_t,4> picks{};
    for(size_t d=0;d<4;d++){ids[d]=order(counts+d*32,32,true);picks[d]=choose(ps,counts+d*32,ids[d],w.tokens);}
    if(policy!="independent"){
      size_t best=ps.size();uint32_t work=UINT32_MAX;
      for(size_t p=0;p<ps.size();p++){
        if(policy=="conservative"&&ps[p].name!="full4")continue;
        bool ok=true;for(size_t d=0;d<4;d++)ok&=safe(ps[p],counts+d*32,ids[d],w.tokens);
        if(ok&&ps[p].work<work){best=p;work=ps[p].work;}
      }
      if(best==ps.size())throw std::runtime_error("No safe common profile");
      picks.fill(best);
    }
    auto decided=Clock::now();
    for(size_t d=0;d<4;d++){
      auto &e=*experts[d];e.shape(w.tokens,ps[picks[d]].tag);e.copy("x",w.x.data(),w.x.size());e.copy("expert_ids",ids[d].data(),128);
      e.copy("route_weights",router.at("route_weights").buf,router.at("route_weights").size);
    }
    auto copied=Clock::now();
    for(auto e:experts){e->bind();e->enqueue();if(serial)e->wait();}
    if(!serial)for(auto e:experts)e->wait();
    auto completed=Clock::now();
    std::array<const uint16_t*,4> partials{};
    for(size_t d=0;d<4;d++)partials[d]=reinterpret_cast<const uint16_t*>(experts[d]->at("y").buf);
    for(size_t j=0;j<w.tokens*2048;j+=8){
      auto sum=_mm256_setzero_ps();
      for(auto ptr:partials)sum=_mm256_add_ps(sum,_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr+j))));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(merged.data()+j),_mm256_cvtps_ph(sum,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC));
    }
    auto end=Clock::now();
    if(!equal(router.at("counts"),w.counts)||!equal(router.at("route_weights"),w.routes))throw std::runtime_error("Router mismatch");
    for(size_t d=0;d<4;d++){
      if(experts[d]->at("counts").size!=128||std::memcmp(experts[d]->at("counts").buf,counts+d*32,128))throw std::runtime_error("Local counts mismatch");
      if(!safe(ps[picks[d]],counts+d*32,ids[d],w.tokens))throw std::runtime_error("Overflow");
    }
    auto ptr=reinterpret_cast<uint8_t*>(merged.data());std::string key=w.name+"_"+policy;
    if(expected.count(key)){if(expected.at(key).size()!=w.x.size()||std::memcmp(expected.at(key).data(),ptr,w.x.size()))throw std::runtime_error("Repeated merged output changed");}
    else{
      expected[key]={ptr,ptr+w.x.size()};QBuffer b{};b.buf=ptr;b.size=w.x.size();writebin(out+"/"+key+"_y.bin",b);
      for(size_t d=0;d<4;d++){writebin(out+"/"+key+"_card"+std::to_string(d)+"_y.bin",experts[d]->at("y"));writebin(out+"/"+key+"_card"+std::to_string(d)+"_ids.bin",experts[d]->at("expert_ids"));}
      writebin(out+"/"+key+"_counts.bin",router.at("counts"));
    }
    if(round>=0){csv<<round<<","<<schedule<<","<<w.name<<","<<policy;for(auto p:picks)csv<<","<<ps[p].tag;
      csv<<","<<ms(begin,end)<<","<<ms(begin,routed)<<","<<ms(routed,decided)<<","<<ms(decided,copied)<<","<<ms(copied,completed)<<","<<ms(completed,end)<<"\n";}
  };
  for(auto &w:ws)for(std::string p:{"independent","common","conservative"})execute(w,p,-1,"warmup");
  for(int round=0;round<3;round++){
    if(round%2)std::reverse(ws.begin(),ws.end());
    for(auto &w:ws)for(std::string p:{"independent","common","conservative"}){
      for(int n=0;n<3;n++)execute(w,p,-1,"warmup");
      for(int n=0;n<N;n++)execute(w,p,round,"held");
    }
    for(int n=0;n<N;n++)for(auto &w:ws)for(std::string p:{"independent","common","conservative"})execute(w,p,round,"cycle");
    for(QID d=0;d<4;d++)s.snapshot("round"+std::to_string(round),d);
  }
  csv.close();if(!csv)throw std::runtime_error("CSV write failed");std::cout<<"MULTICARD_CHECKS_PASSED timed_samples="<<ws.size()*3*N*6<<" serial="<<serial<<"\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<"\n";return 1;}
