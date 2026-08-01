#pragma STDC FENV_ACCESS ON
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
static_assert(std::numeric_limits<float>::is_iec559&&sizeof(float)==4);
static float f(std::uint32_t u){return std::bit_cast<float>(u);}static std::uint32_t u(float x){return std::bit_cast<std::uint32_t>(x);}static float add(float a,float b){volatile float x=a,y=b,z=x+y;return z;}static float sub(float a,float b){volatile float x=a,y=b,z=x-y;return z;}static float mul(float a,float b){volatile float x=a,y=b,z=x*y;return z;}static float divv(float a,float b){volatile float x=a,y=b,z=x/y;return z;}
static std::vector<std::string> split(const std::string&s,char c){std::vector<std::string>v;std::stringstream q(s);std::string x;while(std::getline(q,x,c))v.push_back(x);return v;}static std::uint32_t hx(const std::string&s){return std::stoul(s,nullptr,16);}
int main(int ac,char**av)try{if(ac!=3)throw std::runtime_error("usage: verifier pairs.tsv dots.tsv");if(std::fesetround(FE_TONEAREST))throw std::runtime_error("cannot set RNE");std::ifstream p(av[1]);std::string l;std::getline(p,l);int np=0,nd=0;while(std::getline(p,l)){auto x=split(l,'\t');if(x.size()!=6)throw std::runtime_error("pair row");auto a=f(hx(x[0])),b=f(hx(x[1]));if(u(add(a,b))!=hx(x[2])||u(sub(a,b))!=hx(x[3])||u(mul(a,b))!=hx(x[4])||u(divv(a,b))!=hx(x[5]))throw std::runtime_error("scalar mismatch");++np;}std::ifstream d(av[2]);std::getline(d,l);while(std::getline(d,l)){auto x=split(l,'\t'),aa=split(x[0],','),bb=split(x[1],',');if(x.size()!=3||aa.size()!=64||bb.size()!=64)throw std::runtime_error("dot row");float s=0;for(int i=0;i<64;++i)s=add(s,mul(f(hx(aa[i])),f(hx(bb[i]))));if(u(s)!=hx(x[2]))throw std::runtime_error("dot mismatch");++nd;}std::cout<<"F32_CONTROLLED_REFERENCE_CPP23_OK pairs="<<np<<" ordered_dots="<<nd<<" rounding=RNE contraction=off\n";return 0;}catch(const std::exception&e){std::cerr<<"F32_CONTROLLED_REFERENCE_CPP23_FAIL "<<e.what()<<"\n";return 1;}
