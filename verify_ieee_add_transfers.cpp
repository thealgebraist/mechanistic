#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> v; std::stringstream ss(s); std::string x;
  while (std::getline(ss,x,'\t')) v.push_back(x); return v;
}
int main(int argc,char**argv) try {
  if(argc!=2) throw std::runtime_error("usage: verifier transfers.tsv");
  std::ifstream f(argv[1]); std::string line; std::getline(f,line); int count=0;
  const long double u=std::ldexp(1.0L,-24), eta2=std::ldexp(1.0L,-150);
  while(std::getline(f,line)) {
    auto x=split(line); if(x.size()!=6) throw std::runtime_error("bad row");
    long double l=std::stold(x[1]), r=std::stold(x[2]), scale=std::stold(x[3]);
    auto gain=std::stoull(x[4]), bias=std::stoull(x[5]);
    if(gain!=2) throw std::runtime_error("ADD gain must be 2");
    if(static_cast<long double>(bias)/scale < u*(l+r)+eta2)
      throw std::runtime_error("rounded bias is not conservative");
    ++count;
  }
  if(count!=40) throw std::runtime_error("expected 40 ADD transfers");
  std::cout<<"IEEE_ADD_TRANSFERS_CPP23_OK occurrences="<<count<<"\n";
  return 0;
} catch(const std::exception&e) { std::cerr<<"IEEE_ADD_TRANSFERS_CPP23_FAIL "<<e.what()<<"\n"; return 1; }
