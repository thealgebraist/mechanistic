#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

static std::uint64_t product(const std::string& shape){
  std::stringstream ss(shape);std::string x;std::uint64_t p=1;
  while(std::getline(ss,x,'x'))p*=std::stoull(x);return p;
}

int main(int argc,char**argv){
  assert(argc==3);std::ifstream manifest(argv[1]),blob(argv[2],std::ios::binary|std::ios::ate);
  assert(manifest&&blob);const auto blob_size=static_cast<std::uint64_t>(blob.tellg());
  std::string line;std::getline(manifest,line);std::uint64_t cursor=0;std::map<std::string,int> seen;int blocks=0;
  while(std::getline(manifest,line)){
    std::stringstream ss(line);std::string name,offset,bytes,dtype,shape,hash;
    std::getline(ss,name,'\t');std::getline(ss,offset,'\t');std::getline(ss,bytes,'\t');
    std::getline(ss,dtype,'\t');std::getline(ss,shape,'\t');std::getline(ss,hash);
    const auto off=std::stoull(offset),size=std::stoull(bytes);
    assert(off==cursor&&dtype=="F32BitsLE"&&size==4*product(shape)&&hash.size()==64);
    cursor+=size;++seen[name];++blocks;
  }
  assert(cursor==blob_size);
  for(const char* name:{"whisper.hann_window","whisper.dft_cos","whisper.dft_minus_sin","whisper.mel_weights","carfac.pole_freqs"})assert(seen[name]==1);
  std::cout<<"AUDIO_FILTER_COEFFICIENT_BLOCKS_CPP23_OK blocks="<<blocks<<" bytes="<<blob_size<<"\n";
}
