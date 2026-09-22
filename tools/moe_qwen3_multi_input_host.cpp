// Several routing inputs per resident four-card QPC. Timing excludes setup,
// host input preparation, warmup and output serialization.
#include "QAicApi.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <set>
#include <string>
#include <vector>

#define CK(call) do { auto status = (call); if (status != QS_SUCCESS) { \
  fprintf(stderr, "%s failed: %d\n", #call, int(status)); return 1; } } while (0)
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now()-start).count();
}
struct Input { std::string name; std::vector<uint8_t> data; };

int main(int argc, char **argv) {
  if (argc != 7) {
    fprintf(stderr, "usage: %s QPC_DIR INPUT_MANIFEST OUT_DIR ITERATIONS WARMUP ROUNDS\n", argv[0]);
    return 2;
  }
  const std::string qpcdir=argv[1], outdir=argv[3];
  const int iterations=std::stoi(argv[4]), warmup=std::stoi(argv[5]), rounds=std::stoi(argv[6]);
  if (iterations<1 || warmup<0 || rounds<1 || std::filesystem::exists(outdir)) return 2;
  std::ifstream manifest(argv[2]);
  if (!manifest) return 1;
  std::vector<Input> inputs;
  std::set<std::string> names;
  std::string name, path;
  while (manifest >> name >> path) {
    if (name.empty() || name.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_")!=std::string::npos
        || !names.insert(name).second) return 2;
    std::ifstream file(path,std::ios::binary);
    if (!file) return 1;
    inputs.push_back({name,{std::istreambuf_iterator<char>(file),{}}});
  }
  if (inputs.empty() || !manifest.eof()) return 2;
  std::filesystem::create_directories(outdir);
  const auto setup=Clock::now();
  QID devices[]={0,1,2,3};
  QAicContext *context=nullptr; QAicQueue *queue=nullptr;
  QAicQpcObj *qpc=nullptr; QAicProgram *program=nullptr; QAicExecObj *exec=nullptr;
  CK(qaicCreateContext(&context,nullptr,4,devices,nullptr,nullptr,nullptr,nullptr));
  CK(qaicCreateQueue(context,&queue,nullptr,devices[0]));
  CK(qaicOpenQpcFile(&qpc,(qpcdir+"/programqpc.bin").c_str()));
  QAicProgramProperties_t properties;
  CK(qaicProgramPropertiesInitDefault(&properties));
  properties.devMapping="0:1:2:3";
  CK(qaicCreateProgram(context,&program,&properties,devices[0],"moe_all_active",qpc));
  CK(qaicLoadProgram(program));
  CK(qaicRunActivationCmd(program,QAIC_PROGRAM_CMD_ACTIVATE_FULL));
  QData descriptor{};
  CK(qaicProgramGetIoDescriptor(program,&descriptor));
  QAicExecObjProperties_t ep=QAIC_EXECOBJ_PROPERTIES_DEFAULT;
  CK(qaicCreateExecObj(context,&exec,&ep,program,&descriptor,nullptr,nullptr));
  QAicIoBufferInfo_t *info=nullptr;
  CK(qaicProgramGetIoBufferInfo(program,&info));
  std::vector<QBuffer> buffers(info->numBufferMappings,QBuffer{});
  int input_index=-1;
  for (uint32_t i=0;i<info->numBufferMappings;++i) {
    const auto &mapping=info->bufferMappings[i];
    if (mapping.index>=buffers.size()) return 1;
    auto &buffer=buffers[mapping.index]; buffer.size=mapping.size;
    buffer.buf=static_cast<uint8_t *>(aligned_alloc(4096,(buffer.size+4095)/4096*4096));
    if (!buffer.buf) return 1;
    buffer.type=QBUFFER_TYPE_HEAP; memset(buffer.buf,0,buffer.size);
    if (mapping.ioType==BUFFER_IO_TYPE_INPUT) {
      if (input_index!=-1) return 1;
      input_index=static_cast<int>(mapping.index);
      for (const auto &input:inputs) if (input.data.size()!=buffer.size) return 1;
    }
  }
  if (input_index<0) return 1;
  const double setup_ms=elapsed(setup);
  for (int round=0;round<rounds;++round) {
    for (size_t position=0;position<inputs.size();++position) {
      const auto &input=inputs[round%2 ? inputs.size()-1-position : position];
      memcpy(buffers[input_index].buf,input.data.data(),input.data.size());
      const std::string dest=outdir+"/"+input.name+"/r"+std::to_string(round);
      std::filesystem::create_directories(dest);
      std::vector<double> times;
      for (int iteration=-warmup;iteration<iterations;++iteration) {
        const auto start=Clock::now();
        CK(qaicExecObjSetData(exec,buffers.size(),buffers.data()));
        CK(qaicEnqueueExecObj(queue,exec,nullptr));
        CK(qaicExecObjWaitForCompletion(exec,60000000));
        if (iteration>=0) times.push_back(elapsed(start));
      }
      for (uint32_t i=0;i<info->numBufferMappings;++i) {
        const auto &mapping=info->bufferMappings[i];
        if (mapping.ioType!=BUFFER_IO_TYPE_OUTPUT) continue;
        std::string output_name=mapping.bufferName;
        std::replace(output_name.begin(),output_name.end(),'/','_');
        std::ofstream output(dest+"/"+output_name+".bin",std::ios::binary);
        const auto &buffer=buffers[mapping.index];
        output.write(reinterpret_cast<const char *>(buffer.buf),buffer.size);
        if (!output) return 1;
      }
      std::ofstream timing(dest+"/timing.json");
      timing << std::setprecision(12) << "{\"setup_ms\":" << setup_ms << ",\"warmup\":" << warmup << ",\"samples_ms\":[";
      for (size_t i=0;i<times.size();++i) timing << (i ? "," : "") << times[i];
      timing << "]}\n"; timing.close();
      if (!timing) return 1;
      printf("COMPLETE %s round=%d iterations=%d\n",input.name.c_str(),round,iterations); fflush(stdout);
    }
  }
  CK(qaicReleaseExecObj(exec));
  CK(qaicRunActivationCmd(program,QAIC_PROGRAM_CMD_DEACTIVATE_FULL));
  CK(qaicUnloadProgram(program)); CK(qaicReleaseProgram(program)); CK(qaicCloseQpc(qpc));
  CK(qaicReleaseQueue(queue)); CK(qaicReleaseContext(context));
  for (auto &buffer:buffers) free(buffer.buf);
  return 0;
}
