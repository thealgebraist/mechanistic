#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
static std::vector<std::string> sp(const std::string&s){std::vector<std::string>v;std::stringstream q(s);std::string x;while(std::getline(q,x,'\t'))v.push_back(x);return v;}
struct R{std::uint64_t b,e,rows,cols;};
int main(int ac,char**av)try{if(ac!=4)throw std::runtime_error("usage: verifier model manifest convex.tsv");std::ifstream mf(av[2]);std::string l;std::getline(mf,l);R lm{},nw{};bool hl=false,hn=false;while(std::getline(mf,l)){auto x=sp(l);if(x.size()!=7)continue;R r{std::stoull(x[2]),std::stoull(x[3]),std::stoull(x[4]),std::stoull(x[5])};if(x[0]=="lm_head.weight"){lm=r;hl=true;}if(x[0]=="decoder.final_layer_norm.weight"){nw=r;hn=true;}}if(!hl||!hn)throw std::runtime_error("missing tensors");std::ifstream bin(av[1],std::ios::binary);auto read=[&](R r){bin.seekg(r.b);std::vector<std::uint32_t>v(r.rows*r.cols);bin.read(reinterpret_cast<char*>(v.data()),v.size()*4);return v;};auto a=read(lm),w=read(nw);long double wm=0;for(auto z:w)wm=std::max(wm,std::fabs((long double)std::bit_cast<float>(z)));long double rowmax=0,weighted=0;for(std::uint64_t i=0;i<lm.rows;++i){long double s=0,sw=0;for(std::uint64_t j=0;j<lm.cols;++j){long double z=std::bit_cast<float>(a[i*lm.cols+j]),q=std::bit_cast<float>(w[j]);s+=z*z;sw+=(z*q)*(z*q);}rowmax=std::max(rowmax,std::sqrt(s));weighted=std::max(weighted,std::sqrt(sw));}long double readout=std::sqrt(512.L)*wm,logit=rowmax*readout,wlogit=std::sqrt(512.L)*weighted;std::ifstream cf(av[3]);std::getline(cf,l);std::getline(cf,l);auto x=sp(l);if(x.size()!=9)throw std::runtime_error("bad convex row");if(wm>std::stold(x[0])||rowmax>std::stold(x[1])||readout>std::stold(x[2])||logit>std::stold(x[3])||weighted>std::stold(x[4])||wlogit>std::stold(x[5]))throw std::runtime_error("nonconservative convex certificate");std::cout<<"CONVEX_WEIGHTED_ELLIPSOID_CPP23_OK logit_bound="<<(double)wlogit<<"\n";return 0;}catch(const std::exception&e){std::cerr<<"CONVEX_WEIGHTED_ELLIPSOID_CPP23_FAIL "<<e.what()<<"\n";return 1;}
