#include "moe_joint_runtime.h"
int main(int argc,char **argv)try{
  if(argc!=8)throw std::runtime_error("usage: host ROOT PRECISION OUT EXPERTS ITERATIONS split|fused15|fixed|conservative LAYERS");
  std::string root=argv[1],precision=argv[2],out=argv[3],mode=argv[6];uint32_t E=std::stoul(argv[4]),L=std::stoul(argv[7]);int N=std::stoi(argv[5]);
  bool split=mode=="split";if(N<1||std::filesystem::exists(out))throw std::runtime_error("Invalid output/iterations");
  auto ps=profiles(root,E);auto ws=workloads(root,E,L);std::filesystem::create_directories(out);
  std::ofstream res(out+"/resources.csv"),meta(out+"/packages.csv"),csv(out+"/samples.csv");
  res<<"stage,device,dram_free_kib,nsp_free\n";meta<<"path,device,sdk_constants_size32_sum,required_memory_bytes,cores\n";
  csv<<std::setprecision(9)<<"round,schedule,workload,order,policy,tag,total_ms,router_ms,decision_ms,handoff_ms,bind_ms,expert_ms\n";
  Session s;s.snapshot=[&](const std::string &stage,QID dev){QResourceInfo ri{};CK(qaicGetResourceInfo(dev,&ri));res<<stage<<","<<dev<<","<<ri.dramFree<<","<<unsigned(ri.nspFree)<<"\n";res.flush();};
  s.snapshot("before",0);s.init({0});Program *router=nullptr;
  if(split)router=&s.add(root+"/"+precision+"/router1",0,E,1,meta);
  auto &expert=s.add(root+"/"+precision+"/"+(split?"expert15":mode),0,E,L,meta);s.snapshot("ready",0);
  std::map<std::string,std::vector<uint8_t>> expected;
  auto execute=[&](Workload &w,bool sorted,int forced,int round,const std::string &schedule){
    auto begin=Clock::now();const int32_t *counts=reinterpret_cast<const int32_t*>(w.counts.data());
    if(router){router->shape(w.tokens);router->copy("x",w.x.data(),w.x.size());router->run();counts=reinterpret_cast<const int32_t*>(router->at("counts").buf);}
    auto routed=Clock::now();auto ids=order(counts,E,sorted);size_t pi=forced<0?choose(ps,counts,ids,w.tokens):size_t(forced);
    if(pi>=ps.size()||!safe(ps[pi],counts,ids,w.tokens))throw std::runtime_error("Unsafe/unlisted dispatch");
    std::vector<std::vector<int32_t>> layerids={ids};
    for(uint32_t l=1;l<L;l++){layerids.push_back(order(counts+l*E,E,sorted));if(!safe(ps[pi],counts+l*E,layerids.back(),w.tokens))throw std::runtime_error("Unsafe later layer");}
    auto decided=Clock::now();expert.shape(w.tokens,ps[pi].tag);expert.copy("x",w.x.data(),w.x.size());
    for(uint32_t l=0;l<L;l++)expert.copy(L==1?"expert_ids":"expert_ids"+std::to_string(l),layerids[l].data(),E*4);
    if(router)expert.copy("route_weights",router->at("route_weights").buf,router->at("route_weights").size);
    auto copied=Clock::now();expert.bind();auto bound=Clock::now();expert.enqueue();expert.wait();auto end=Clock::now();
    if(!equal(expert.at("counts"),w.counts))throw std::runtime_error("Count mismatch "+w.name);
    if(router&&(!equal(router->at("counts"),w.counts)||!equal(router->at("route_weights"),w.routes)))throw std::runtime_error("Router mismatch "+w.name);
    std::string key=w.name+"_"+(sorted?"sorted":"identity")+"_p"+std::to_string(ps[pi].tag);
    if(expected.count(key)){if(!equal(expert.at("y"),expected.at(key)))throw std::runtime_error("Repeated output mismatch");}
    else{auto &b=expert.at("y");expected[key]={b.buf,b.buf+b.size};writebin(out+"/"+key+"_y.bin",b);writebin(out+"/"+key+"_counts.bin",expert.at("counts"));}
    if(round>=0)csv<<round<<","<<schedule<<","<<w.name<<","<<(sorted?"sorted":"identity")<<","<<(forced<0?"current":"forced")<<","<<ps[pi].tag<<","<<ms(begin,end)<<","<<ms(begin,routed)<<","<<ms(routed,decided)<<","<<ms(decided,copied)<<","<<ms(copied,bound)<<","<<ms(bound,end)<<"\n";
  };
  struct Case{size_t w;bool sorted;int p;};std::vector<Case> cases;
  for(size_t wi=0;wi<ws.size();wi++)for(bool sorted:{false,true}){
    auto &w=ws[wi];auto counts=reinterpret_cast<const int32_t*>(w.counts.data());auto ids=order(counts,E,sorted);
    for(size_t pi=0;pi<ps.size();pi++){
      auto &p=ps[pi];if(!safe(p,counts,ids,w.tokens))continue;
      bool all=true;for(uint32_t l=1;l<L;l++)all&=safe(p,counts+l*E,order(counts+l*E,E,sorted),w.tokens);if(!all)continue;
      if(mode=="fixed"&&(p.name!="medium4"||w.tokens!=128))continue;
      if(mode=="conservative"&&(p.name!="full4"||w.tokens!=128))continue;
      cases.push_back({wi,sorted,int(pi)});
    }
    if(split&&sorted)cases.push_back({wi,true,-1});
  }
  for(auto c:cases)execute(ws[c.w],c.sorted,c.p,-1,"warmup");
  for(int round=0;round<3;round++){
    if(round%2)std::reverse(cases.begin(),cases.end());
    for(auto c:cases){for(int n=0;n<3;n++)execute(ws[c.w],c.sorted,c.p,-1,"warmup");for(int n=0;n<N;n++)execute(ws[c.w],c.sorted,c.p,round,"held");}
    for(int n=0;n<N;n++)for(auto c:cases)execute(ws[c.w],c.sorted,c.p,round,"cycle");
    s.snapshot("round"+std::to_string(round),0);
  }
  csv.close();if(!csv)throw std::runtime_error("CSV write failed");std::cout<<"JOINT_CHECKS_PASSED cases="<<cases.size()<<" timed_samples="<<cases.size()*N*6<<"\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<"\n";return 1;}
