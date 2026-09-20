#include "QAicApi.h"
#include <cstdio>
#include <string>
int main(int argc,char**argv){
  for(int i=1;i<argc;i++){
    QAicQpcObj* q=nullptr; std::string p=std::string(argv[i])+"/programqpc.bin";
    if(qaicOpenQpcFile(&q,p.c_str())!=QS_SUCCESS){printf("%s: open FAIL\n",argv[i]);continue;}
    const QAicQpcInfo_t* info=nullptr;
    if(qaicQpcGetInfo(q,&info)!=QS_SUCCESS){printf("%s: getInfo FAIL\n",argv[i]);continue;}
    printf("%s\n  numPrograms=%u numConstants=%u\n",argv[i],info->numPrograms,info->numConstants);
    for(uint32_t k=0;k<info->numPrograms;k++){
      auto&g=info->programInfo[k];
      printf("   [%u] name='%s' idx=%u cores=%u batch=%d mem=%.1f MB  IO buffers=%u\n",
        k,g.name,g.programIndex,g.numCores,g.batchSize,g.reqMem/1048576.0,g.ioBufferInfo.numBufferMappings);
    }
    qaicCloseQpc(q);
  }
  return 0;
}
